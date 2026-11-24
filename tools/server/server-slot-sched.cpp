#include "server-slot-sched.h"

#include "common.h"
#include "hash/hash.h"
#include "llama.h"
#include "log.h"

#include <algorithm>
#include <climits>
#include <fstream>

//
// server_prompt
//

server_prompt_reuse server_prompt::get_reuse(int32_t n_prefix, bool has_new_tokens, int32_t n_swa, llama_pos pos_min) const {
    server_prompt_reuse res;
    res.n_past   = n_prefix;
    res.pos_next = tokens.pos_next(n_prefix);

    // the first evaluated token attends to the positions > pos_eval - n_swa, so a SWA state
    // starting at pos_eval - n_swa + 1 is enough, while a state without SWA must end before pos_eval
    const llama_pos pos_eval = res.pos_next - (has_new_tokens ? 0 : 1);

    // a state or a checkpoint is useful when its pos_min is below this threshold
    const auto pos_min_thold = std::max(0, n_swa > 0 ? pos_eval - n_swa + 2 : pos_eval);

    if (pos_min < pos_min_thold) {
        return res;
    }

    // search for a context checkpoint
    const auto it = std::find_if(
        checkpoints.rbegin(),
        checkpoints.rend(),
        [&](const auto & ckpt) {
            const auto & cur = *ckpt;
            // guarantee that a checkpoint will result in at least one token being processed [TAG_PROMPT_LOGITS]
            SRV_TRC("checking checkpoint with [%d, %d] against %d...\n", cur.pos_min, cur.pos_max, pos_min_thold);
            // workaround for [TAG_CHECKPOINTS_FIX_POS_MIN]
            if (cur.pos_max > res.pos_next) {
                return false;
            }
            return cur.pos_min < pos_min_thold || cur.pos_min == 0;
        }
    );

    if (it == checkpoints.rend()) {
        res.n_past   = 0;
        res.pos_next = 0;
        return res;
    }

    const auto & cur = **it;

    res.pos_next = std::min(res.pos_next, std::max(cur.pos_min + 1, cur.pos_max));
    res.n_past   = std::min(tokens.size_up_to_pos(res.pos_next), (size_t) cur.n_tokens);
    res.ckpt     = *it;

    return res;
}

//
// server_prompt_cache_disk
//

namespace fs = std::filesystem;

static const char * FILE_EXT     = ".kvc";
static const char * FILE_EXT_TMP = ".tmp";

enum server_cache_disk_flag : uint64_t {
    SERVER_CACHE_DISK_FLAG_PINNED = 1ULL << 0, // core/base prompt: never evicted by size limit or replaced by a longer prompt
};

struct server_cache_disk_header {
    uint32_t magic   = 0x3243564b; // "KVC2"
    uint32_t version = 2;

    uint64_t flags         = 0;
    uint64_t n_tokens      = 0;
    uint64_t n_hashes      = 0;
    uint64_t n_checkpoints = 0;

    // in bytes
    uint64_t size_tokens      = 0; // server_tokens::serialize()
    uint64_t size_main        = 0;
    uint64_t size_drft        = 0;
    uint64_t size_checkpoints = 0; // headers included

    server_tokens::prefix_hash hash = {};
};
static_assert(sizeof(server_cache_disk_header) == 104, "unexpected header size");

struct server_cache_disk_header_v1 {
    uint32_t magic   = 0x3243564b; // "KVC2"
    uint32_t version = 1;

    uint64_t n_tokens      = 0;
    uint64_t n_hashes      = 0;
    uint64_t n_checkpoints = 0;

    uint64_t size_tokens      = 0;
    uint64_t size_main        = 0;
    uint64_t size_drft        = 0;
    uint64_t size_checkpoints = 0;

    server_tokens::prefix_hash hash = {};
};
static_assert(sizeof(server_cache_disk_header_v1) == 96, "unexpected header v1 size");

struct server_cache_disk_checkpoint {
    int64_t n_tokens = 0;
    int32_t pos_min  = 0;
    int32_t pos_max  = 0;

    // in bytes
    uint64_t size_tgt  = 0;
    uint64_t size_dft  = 0;
    uint64_t size_spec = 0;
};
static_assert(sizeof(server_cache_disk_checkpoint) == 40, "unexpected checkpoint header size");

static std::string make_name(const std::string & compat, const server_tokens::prefix_hash & hash) {
    static const char hex[] = "0123456789abcdef";

    // half of the hash is enough for a file name
    std::string res = compat + "-";
    for (size_t i = 0; i < hash.size() / 2; i++) {
        res += hex[hash[i] >> 4];
        res += hex[hash[i] & 0xf];
    }
    return res + FILE_EXT;
}

static void read_bytes(std::ifstream & in, void * data, size_t size) {
    in.read((char *) data, size);
    if (!in.good()) {
        throw std::runtime_error("unexpected end of file");
    }
}

static void write_bytes(std::ofstream & out, const void * data, size_t size) {
    out.write((const char *) data, size);
}

static server_cache_disk_header header_from_v1(const server_cache_disk_header_v1 & v1) {
    server_cache_disk_header header;
    header.magic           = v1.magic;
    header.version         = 2;
    header.flags           = 0;
    header.n_tokens        = v1.n_tokens;
    header.n_hashes        = v1.n_hashes;
    header.n_checkpoints   = v1.n_checkpoints;
    header.size_tokens     = v1.size_tokens;
    header.size_main       = v1.size_main;
    header.size_drft       = v1.size_drft;
    header.size_checkpoints = v1.size_checkpoints;
    header.hash            = v1.hash;
    return header;
}

// reads the header and the hash chain, the stream is left at the start of the tokens
static server_cache_disk_header read_header(std::ifstream & in, const std::string & compat, server_prompt_cache_disk_entry & entry) {
    uint32_t magic   = 0;
    uint32_t version = 0;
    read_bytes(in, &magic, sizeof(magic));
    read_bytes(in, &version, sizeof(version));

    server_cache_disk_header header;
    if (magic != server_cache_disk_header().magic) {
        throw std::runtime_error("unsupported file version");
    }

    if (version == 1) {
        server_cache_disk_header_v1 v1;
        v1.magic   = magic;
        v1.version = version;
        read_bytes(in, reinterpret_cast<char *>(&v1) + sizeof(magic) + sizeof(version), sizeof(v1) - sizeof(magic) - sizeof(version));
        header = header_from_v1(v1);
    } else if (version == 2) {
        header.magic   = magic;
        header.version = version;
        read_bytes(in, reinterpret_cast<char *>(&header) + sizeof(magic) + sizeof(version), sizeof(header) - sizeof(magic) - sizeof(version));
    } else {
        throw std::runtime_error("unsupported file version");
    }

    const uint64_t size_hashes = header.n_hashes * sizeof(server_tokens::prefix_hash);
    const uint64_t size_file   = sizeof(header) + size_hashes + header.size_tokens + header.size_main + header.size_drft + header.size_checkpoints;

    if (header.n_hashes != header.n_tokens / server_tokens::hash_block_size ||
            header.size_tokens % sizeof(llama_token) != 0 ||
            size_file != entry.size ||
            make_name(compat, header.hash) != entry.name) {
        throw std::runtime_error("invalid header");
    }

    entry.n_tokens = header.n_tokens;
    entry.hash     = header.hash;
    entry.pinned   = (header.flags & SERVER_CACHE_DISK_FLAG_PINNED) != 0;
    entry.hashes.resize(header.n_hashes);
    if (size_hashes > 0) {
        read_bytes(in, entry.hashes.data(), size_hashes);
    }

    return header;
}

server_prompt_cache_disk::server_prompt_cache_disk(const std::string & dir, const std::string & compat, size_t limit_size, bool has_mtmd, int32_t n_threads)
        : dir(dir), compat(compat), limit_size(limit_size), has_mtmd(has_mtmd), n_workers((size_t) std::max<int32_t>(1, n_threads)) {
    scan();

    SRV_TRC("disk prompt cache: %zu background writer(s)\n", n_workers);
}

server_prompt_cache_disk::~server_prompt_cache_disk() {
    {
        std::lock_guard<std::mutex> lock(mutex);
        stopping = true;

        // the queued prompts are a cache, drop them instead of delaying the server exit
        queue.clear();
        queued.clear();
    }
    cv_work.notify_all();
    for (auto & worker : workers) {
        worker.join();
    }
    if (warmup_thread.joinable()) {
        warmup_thread.join();
    }
}

void server_prompt_cache_disk::start_warmup() {
    if (warmup_thread.joinable()) {
        return;
    }

    std::vector<std::string> names;
    {
        std::lock_guard<std::mutex> lock(mutex);
        for (auto it = entries.rbegin(); it != entries.rend(); ++it) {
            if (it->n_tokens > 0) {
                names.push_back(it->name);
            }
        }
    }

    if (names.empty()) {
        return;
    }

    warmup_thread = std::thread([this, names = std::move(names)]() {
        std::vector<char> buf(4 * 1024 * 1024);
        for (const auto & name : names) {
            if (stopping.load(std::memory_order_relaxed)) {
                return;
            }

            std::ifstream in(dir / name, std::ios::binary);
            if (!in.good()) {
                continue;
            }

            while (in.good() && !stopping.load(std::memory_order_relaxed)) {
                in.read(buf.data(), buf.size());
            }

            SRV_TRC("warmed disk prompt cache entry '%s'\n", name.c_str());
        }
    });
}

void server_prompt_cache_disk::worker_loop() {
    while (true) {
        job cur;
        {
            std::unique_lock<std::mutex> lock(mutex);
            cv_work.wait(lock, [this]() { return stopping.load() || !queue.empty(); });
            if (stopping.load()) {
                return;
            }
            cur = std::move(queue.front());
            queue.pop_front();
            n_active++;
        }

        try {
            // a write in progress is interrupted at the next chunk when the server shuts down
            save_impl(*cur.state, cur.pinned, true);
        } catch (const std::exception & e) {
            SRV_ERR("failed to save prompt to disk: %s\n", e.what());
        }

        {
            std::lock_guard<std::mutex> lock(mutex);
            queued.erase(cur.name);
            n_active--;
        }
        cv_done.notify_all();
    }
}

void server_prompt_cache_disk::enqueue(std::shared_ptr<const server_prompt_cache_state> state, bool pinned) {
    if (!state || state->prompt.tokens.empty() || state->data.main.empty() || state->prompt.lora_changed) {
        return;
    }

    const std::string name = make_name(compat, state->prompt.tokens.get_hash(state->prompt.tokens.size()));

    std::lock_guard<std::mutex> lock(mutex);

    if (queued.count(name) > 0 || writing.count(name) > 0) {
        return;
    }
    for (const auto & entry : entries) {
        if (entry.name == name) {
            return;
        }
    }

    if (workers.empty()) {
        for (size_t i = 0; i < n_workers; i++) {
            workers.emplace_back([this]() { worker_loop(); });
        }
    }

    queued.insert(name);
    queue.push_back({ std::move(state), name, pinned });
    cv_work.notify_one();
}

void server_prompt_cache_disk::wait() {
    std::unique_lock<std::mutex> lock(mutex);
    cv_done.wait(lock, [this]() { return queue.empty() && n_active == 0; });
}

bool server_prompt_cache_disk::busy(const server_prompt_cache_state & state) const {
    if (state.prompt.tokens.empty()) {
        return false;
    }

    const std::string name = make_name(compat, state.prompt.tokens.get_hash(state.prompt.tokens.size()));

    std::lock_guard<std::mutex> lock(mutex);
    return queued.count(name) > 0 || writing.count(name) > 0;
}

bool server_prompt_cache_disk::has(const server_tokens & tokens) const {
    if (tokens.empty()) {
        return false;
    }

    const std::string name = make_name(compat, tokens.get_hash(tokens.size()));

    std::lock_guard<std::mutex> lock(mutex);
    if (queued.count(name) > 0 || writing.count(name) > 0) {
        return true;
    }

    return std::any_of(entries.begin(), entries.end(), [&name](const auto & entry) {
        return entry.name == name;
    });
}

std::vector<server_prompt_cache_disk_entry> server_prompt_cache_disk::snapshot() const {
    std::lock_guard<std::mutex> lock(mutex);
    return std::vector<server_prompt_cache_disk_entry>(entries.begin(), entries.end());
}

void server_prompt_cache_disk::scan() {
    size_t n_usable = 0;

    std::error_code ec;
    for (const auto & file : fs::directory_iterator(dir, ec)) {
        if (!file.is_regular_file(ec)) {
            continue;
        }

        const std::string name = file.path().filename().string();
        const bool is_compat   = string_starts_with(name, compat + "-");

        // left by a server that was killed while saving
        if (is_compat && string_ends_with(name, FILE_EXT_TMP)) {
            fs::remove(file.path(), ec);
            continue;
        }

        if (!string_ends_with(name, FILE_EXT)) {
            continue;
        }

        server_prompt_cache_disk_entry entry;
        entry.name   = name;
        entry.size   = file.file_size(ec);
        entry.t_used = file.last_write_time(ec);

        if (is_compat) {
            try {
                std::ifstream in(file.path(), std::ios::binary);
                read_header(in, compat, entry);
            } catch (const std::exception & e) {
                SRV_WRN("removing '%s': %s\n", name.c_str(), e.what());
                fs::remove(file.path(), ec);
                continue;
            }
            n_usable++;
        }

        entries.push_back(std::move(entry));
    }

    entries.sort([](const auto & a, const auto & b) {
        return a.t_used < b.t_used;
    });

    SRV_INF("disk prompt cache '%s': %zu prompts, %zu files of other models, %.3f MiB (limit: %.3f MiB)\n",
            dir.string().c_str(), n_usable, entries.size() - n_usable, size() / (1024.0 * 1024.0), limit_size / (1024.0 * 1024.0));
}

size_t server_prompt_cache_disk::size() const {
    std::lock_guard<std::mutex> lock(mutex);
    return size_unlocked();
}

size_t server_prompt_cache_disk::size_unlocked() const {
    size_t res = writing_size;

    for (const auto & entry : entries) {
        res += entry.size;
    }

    return res;
}

server_prompt_cache_disk::entry_list::iterator server_prompt_cache_disk::remove_unlocked(entry_list::iterator it) {
    std::error_code ec;
    fs::remove(dir / it->name, ec);

    return entries.erase(it);
}

void server_prompt_cache_disk::touch_unlocked(entry_list::iterator it) {
    std::error_code ec;
    it->t_used = fs::file_time_type::clock::now();
    fs::last_write_time(dir / it->name, it->t_used, ec);

    entries.splice(entries.end(), entries, it);
}

void server_prompt_cache_disk::pin_unlocked(entry_list::iterator it) {
    if (it->pinned) {
        return;
    }

    const fs::path path = dir / it->name;
    std::fstream io(path, std::ios::binary | std::ios::in | std::ios::out);
    if (!io.good()) {
        SRV_WRN("failed to pin '%s': cannot open file\n", it->name.c_str());
        return;
    }

    uint32_t magic = 0;
    uint32_t version = 0;
    uint64_t flags = 0;
    io.read((char *) &magic, sizeof(magic));
    io.read((char *) &version, sizeof(version));
    io.read((char *) &flags, sizeof(flags));
    if (!io.good() || magic != server_cache_disk_header().magic || version != 2) {
        SRV_WRN("failed to pin '%s': unsupported file header\n", it->name.c_str());
        return;
    }

    flags |= SERVER_CACHE_DISK_FLAG_PINNED;
    io.clear();
    io.seekp((std::streamoff)(sizeof(magic) + sizeof(version)), std::ios::beg);
    io.write((const char *) &flags, sizeof(flags));
    io.flush();
    if (!io.good()) {
        SRV_WRN("failed to pin '%s': cannot write flags\n", it->name.c_str());
        return;
    }

    it->pinned = true;
    SRV_INF("pinned disk prompt cache entry '%s'\n", it->name.c_str());
}

void server_prompt_cache_disk::pin(const server_prompt_cache_disk_entry & entry) {
    std::lock_guard<std::mutex> lock(mutex);

    const auto it = std::find_if(entries.begin(), entries.end(), [&entry](const auto & cur) {
        return cur.name == entry.name;
    });
    if (it == entries.end()) {
        return;
    }

    pin_unlocked(it);
}

bool server_prompt_cache_disk::save(const server_prompt_cache_state & state, bool pinned) {
    return save_impl(state, pinned, false);
}

bool server_prompt_cache_disk::save_impl(const server_prompt_cache_state & state, bool pinned, bool cancelable) {
    const auto & tokens = state.prompt.tokens;

    if (tokens.empty() || state.data.main.empty()) {
        return false;
    }

    server_cache_disk_header header;
    header.flags = pinned ? (uint64_t) SERVER_CACHE_DISK_FLAG_PINNED : 0;
    header.hash  = tokens.get_hash(tokens.size());

    const std::string name = make_name(compat, header.hash);

    std::vector<char> tokens_data;
    try {
        tokens_data = tokens.serialize();
    } catch (const std::exception & e) {
        SRV_WRN("failed to serialize the prompt: %s\n", e.what());
        return false;
    }

    const auto & hashes = tokens.get_prefix_hashes();

    header.n_tokens      = tokens.size();
    header.n_hashes      = hashes.size();
    header.n_checkpoints = state.prompt.checkpoints.size();
    header.size_tokens   = tokens_data.size();
    header.size_main     = state.data.main.size();
    header.size_drft     = state.data.drft.size();

    for (const auto & ckpt : state.prompt.checkpoints) {
        header.size_checkpoints += sizeof(server_cache_disk_checkpoint) + ckpt->data_tgt.size() + ckpt->data_dft.size() + ckpt->data_spec.size();
    }

    const size_t size_hashes = hashes.size() * sizeof(server_tokens::prefix_hash);
    const size_t size_file   = sizeof(header) + size_hashes + header.size_tokens + header.size_main + header.size_drft + header.size_checkpoints;

    std::unique_lock<std::mutex> lock(mutex);

    if (writing.count(name) > 0) {
        return false;
    }

    for (auto it = entries.begin(); it != entries.end();) {
        if (it->name == name) {
            SRV_TRC(" - prompt is already on disk, '%s'\n", name.c_str());
            if ((pinned || pin_core) && !it->pinned) {
                pin_unlocked(it);
            }
            touch_unlocked(it);
            return true;
        }

        if (0 < it->n_tokens && it->n_tokens < tokens.size() && tokens.get_hash(it->n_tokens) == it->hash) {
            if (it->pinned) {
                ++it;
            } else if (pin_core) {
                SRV_TRC(" - keeping prefix prompt as pinned core on disk, '%s'\n", it->name.c_str());
                pin_unlocked(it);
                ++it;
            } else if (writing.count(it->name) == 0) {
                SRV_TRC(" - removing obsolete prompt with length %zu from disk, '%s'\n", it->n_tokens, it->name.c_str());
                it = remove_unlocked(it);
            } else {
                ++it;
            }
        } else {
            ++it;
        }
    }

    if (limit_size > 0) {
        if (size_file > limit_size) {
            SRV_WRN(" - prompt size %.3f MiB exceeds disk cache size limit %.3f MiB, skipping\n",
                    size_file / (1024.0 * 1024.0), limit_size / (1024.0 * 1024.0));
            return false;
        }

        while (size_unlocked() + size_file > limit_size) {
            auto it = std::find_if(entries.begin(), entries.end(), [this](const auto & e) {
                return !e.pinned && writing.count(e.name) == 0;
            });
            if (it == entries.end()) {
                SRV_WRN(" - disk cache size limit %.3f MiB reached, but all entries are pinned; keeping '%s'\n",
                        limit_size / (1024.0 * 1024.0), name.c_str());
                break;
            }
            SRV_TRC(" - making room on disk, removing oldest non-pinned file '%s' (size = %.3f MiB)\n",
                    it->name.c_str(), it->size / (1024.0 * 1024.0));
            remove_unlocked(it);
        }
    }

    writing.insert(name);
    writing_size += size_file;
    lock.unlock();

    const fs::path path     = dir / name;
    const fs::path path_tmp = dir / (name + FILE_EXT_TMP);

    bool ok        = true;
    bool cancelled = false;

    std::error_code ec;
    {
        // chunks let a shutdown interrupt a multi-hundred-MiB state within tens of milliseconds
        static constexpr size_t chunk_size = 4 * 1024 * 1024;

        // a large buffer keeps the writes of a multi-hundred-MiB state in few syscalls
        std::vector<char> io_buf(1024 * 1024);

        std::ofstream out;
        out.rdbuf()->pubsetbuf(io_buf.data(), io_buf.size());
        out.open(path_tmp, std::ios::binary | std::ios::trunc);

        const auto write_chunked = [&](const void * data, size_t size) {
            const char * ptr = (const char *) data;

            while (size > 0) {
                if (cancelable && stopping.load(std::memory_order_relaxed)) {
                    cancelled = true;
                    ok        = false;
                    return;
                }

                const size_t n = std::min(size, chunk_size);
                write_bytes(out, ptr, n);
                if (!out.good()) {
                    ok = false;
                    return;
                }

                ptr  += n;
                size -= n;
            }
        };

        write_chunked(&header, sizeof(header));
        write_chunked(hashes.data(), size_hashes);
        write_chunked(tokens_data.data(), tokens_data.size());
        write_chunked(state.data.main.data(), state.data.main.size());
        write_chunked(state.data.drft.data(), state.data.drft.size());

        for (const auto & ckpt : state.prompt.checkpoints) {
            if (!ok) {
                break;
            }

            server_cache_disk_checkpoint ckpt_header;
            ckpt_header.n_tokens  = ckpt->n_tokens;
            ckpt_header.pos_min   = ckpt->pos_min;
            ckpt_header.pos_max   = ckpt->pos_max;
            ckpt_header.size_tgt  = ckpt->data_tgt.size();
            ckpt_header.size_dft  = ckpt->data_dft.size();
            ckpt_header.size_spec = ckpt->data_spec.size();

            write_chunked(&ckpt_header, sizeof(ckpt_header));
            write_chunked(ckpt->data_tgt.data(), ckpt->data_tgt.size());
            write_chunked(ckpt->data_dft.data(), ckpt->data_dft.size());
            write_chunked(ckpt->data_spec.data(), ckpt->data_spec.size());
        }

        out.flush();
        if (ok && !out.good()) {
            ok = false;
        }

        out.close();
    }

    if (!ok) {
        if (cancelled) {
            SRV_INF(" - save cancelled, the server is shutting down, '%s'\n", name.c_str());
        } else {
            SRV_ERR("failed to write '%s'\n", path_tmp.string().c_str());
        }
        fs::remove(path_tmp, ec);
    }

    if (ok) {
        // a reader never sees a half written file
        fs::rename(path_tmp, path, ec);
        if (ec) {
            SRV_ERR("failed to rename '%s': %s\n", path_tmp.string().c_str(), ec.message().c_str());
            fs::remove(path_tmp, ec);
            ok = false;
        }
    }

    bool already = false;
    bool added   = false;

    if (ok) {
        server_prompt_cache_disk_entry entry;
        entry.name     = name;
        entry.size     = size_file;
        entry.t_used   = fs::last_write_time(path, ec);
        entry.n_tokens = tokens.size();
        entry.hash     = header.hash;
        entry.hashes   = hashes;
        entry.pinned   = pinned;

        std::lock_guard<std::mutex> lock_commit(mutex);
        writing.erase(name);
        writing_size -= size_file;

        // another writer may have saved this prompt or a prefix while this file was written
        for (auto it = entries.begin(); it != entries.end();) {
            if (it->name == name) {
                touch_unlocked(it);
                already = true;
                break;
            }

            if (0 < it->n_tokens && it->n_tokens < tokens.size() && tokens.get_hash(it->n_tokens) == it->hash &&
                    !it->pinned && writing.count(it->name) == 0) {
                it = remove_unlocked(it);
            } else {
                ++it;
            }
        }

        if (!already) {
            entries.push_back(std::move(entry));
            added = true;
        }
    } else {
        std::lock_guard<std::mutex> lock_commit(mutex);
        writing.erase(name);
        writing_size -= size_file;
    }
    cv_done.notify_all();

    if (!ok) {
        return false;
    }

    if (already) {
        return true;
    }

    if (added) {
        SRV_INF("saved prompt with %zu tokens to disk, %.3f MiB, '%s'\n", tokens.size(), size_file / (1024.0 * 1024.0), name.c_str());
    }

    return true;
}

bool server_prompt_cache_disk::load(const server_prompt_cache_disk_entry & entry, server_prompt_cache_state & state) {
    std::ifstream in;
    server_prompt_cache_disk_entry cur;

    {
        std::lock_guard<std::mutex> lock(mutex);

        const auto it = std::find_if(entries.begin(), entries.end(), [&entry](const auto & e) {
            return e.name == entry.name;
        });
        if (it == entries.end()) {
            return false;
        }

        cur = *it;

        // open while the file cannot be removed by a background save
        in.open(dir / cur.name, std::ios::binary);
        if (!in.good()) {
            SRV_WRN("removing '%s': cannot open file\n", cur.name.c_str());
            remove_unlocked(it);
            return false;
        }
    }

    server_prompt_cache_state res;

    try {
        const auto header = read_header(in, compat, cur);

        llama_tokens tokens_data(header.size_tokens / sizeof(llama_token));
        read_bytes(in, tokens_data.data(), header.size_tokens);

        res.prompt.tokens = server_tokens::deserialize(tokens_data, has_mtmd);

        if (res.prompt.tokens.size() != cur.n_tokens || res.prompt.tokens.get_hash(cur.n_tokens) != cur.hash) {
            throw std::runtime_error("the tokens do not match the hash");
        }

        res.data.main.resize(header.size_main);
        res.data.drft.resize(header.size_drft);

        const std::streampos off_main = in.tellg();
        const std::streampos off_drft = off_main + (std::streamoff) header.size_main;
        const std::streampos off_ckpt = off_drft + (std::streamoff) header.size_drft;

        const fs::path path = dir / cur.name;
        const size_t   n_read_threads = std::max<size_t>(1, std::min<size_t>(n_workers, 4));

        auto read_blob = [&](std::streampos off, void * dst, size_t size) {
            if (size == 0) {
                return;
            }

            const size_t n_thr = size >= 32 * 1024 * 1024 ? n_read_threads : 1;
            if (n_thr == 1) {
                std::ifstream in_blob(path, std::ios::binary);
                in_blob.seekg(off);
                read_bytes(in_blob, dst, size);
                return;
            }

            std::atomic<bool> failed{false};
            std::string err;
            std::mutex err_mutex;
            std::vector<std::thread> readers;
            readers.reserve(n_thr);

            const size_t chunk = (size + n_thr - 1) / n_thr;
            for (size_t i = 0; i < n_thr; i++) {
                const size_t start = i * chunk;
                if (start >= size) {
                    break;
                }
                const size_t len = std::min(chunk, size - start);
                readers.emplace_back([&, start, len]() {
                    if (failed.load(std::memory_order_relaxed)) {
                        return;
                    }
                    try {
                        std::ifstream in_blob(path, std::ios::binary);
                        in_blob.seekg(off + (std::streamoff) start);
                        read_bytes(in_blob, (char *) dst + start, len);
                    } catch (const std::exception & e) {
                        bool expected = false;
                        if (failed.compare_exchange_strong(expected, true)) {
                            std::lock_guard<std::mutex> lock(err_mutex);
                            err = e.what();
                        }
                    }
                });
            }

            for (auto & t : readers) {
                t.join();
            }

            if (failed.load()) {
                throw std::runtime_error(err);
            }
        };

        read_blob(off_main, res.data.main.data(), header.size_main);
        read_blob(off_drft, res.data.drft.data(), header.size_drft);

        in.seekg(off_ckpt);
        uint64_t size_checkpoints = 0;
        for (uint64_t i = 0; i < header.n_checkpoints; i++) {
            server_cache_disk_checkpoint ckpt_header;
            read_bytes(in, &ckpt_header, sizeof(ckpt_header));

            size_checkpoints += sizeof(ckpt_header) + ckpt_header.size_tgt + ckpt_header.size_dft + ckpt_header.size_spec;
            if (size_checkpoints > header.size_checkpoints) {
                throw std::runtime_error("invalid checkpoint size");
            }

            auto ckpt = std::make_shared<common_prompt_checkpoint>();
            ckpt->n_tokens = ckpt_header.n_tokens;
            ckpt->pos_min  = ckpt_header.pos_min;
            ckpt->pos_max  = ckpt_header.pos_max;
            ckpt->data_tgt.resize(ckpt_header.size_tgt);
            ckpt->data_dft.resize(ckpt_header.size_dft);
            ckpt->data_spec.resize(ckpt_header.size_spec);
            read_bytes(in, ckpt->data_tgt.data(), ckpt_header.size_tgt);
            read_bytes(in, ckpt->data_dft.data(), ckpt_header.size_dft);
            read_bytes(in, ckpt->data_spec.data(), ckpt_header.size_spec);

            res.prompt.checkpoints.push_back(std::move(ckpt));
        }
    } catch (const std::bad_alloc & e) {
        SRV_ERR("failed to allocate memory for '%s': %s\n", cur.name.c_str(), e.what());
        return false;
    } catch (const std::exception & e) {
        SRV_WRN("removing '%s': %s\n", cur.name.c_str(), e.what());

        std::lock_guard<std::mutex> lock(mutex);
        const auto it = std::find_if(entries.begin(), entries.end(), [&cur](const auto & e) {
            return e.name == cur.name;
        });
        if (it != entries.end()) {
            remove_unlocked(it);
        }
        return false;
    }

    {
        std::lock_guard<std::mutex> lock(mutex);
        const auto it = std::find_if(entries.begin(), entries.end(), [&cur](const auto & e) {
            return e.name == cur.name;
        });
        if (it != entries.end()) {
            touch_unlocked(it);
        }
    }

    SRV_INF("loaded prompt with %zu tokens from disk, %.3f MiB, '%s'\n", cur.n_tokens, cur.size / (1024.0 * 1024.0), cur.name.c_str());

    state = std::move(res);

    return true;
}

template <typename T>
static void compat_add(std::string & buf, const T & value) {
    static_assert(std::is_trivially_copyable<T>::value, "T must be trivially copyable");
    buf.append((const char *) &value, sizeof(value));
}

static void compat_add(std::string & buf, const std::string & value) {
    compat_add(buf, (uint64_t) value.size());
    buf += value;
}

// a file is known by its path, size and time, reading a model to hash it takes too long
static void compat_add_file(std::string & buf, const std::string & path) {
    std::error_code ec;
    compat_add(buf, path);
    compat_add(buf, (uint64_t) (path.empty() ? 0 : fs::file_size(path, ec)));
    compat_add(buf, (int64_t)  (path.empty() ? 0 : fs::last_write_time(path, ec).time_since_epoch().count()));
}

std::string server_prompt_cache_disk_compat(const common_params & params) {
    std::string buf;

    compat_add(buf, (uint32_t) LLAMA_STATE_SEQ_VERSION);
    compat_add(buf, (uint64_t) server_tokens::hash_block_size);

    compat_add_file(buf, params.model.path);
    compat_add_file(buf, params.mmproj.path);
    compat_add_file(buf, params.speculative.draft.mparams.path);

    for (const auto & lora : params.lora_adapters) {
        compat_add_file(buf, lora.path);
        compat_add(buf, lora.scale);
    }

    for (const auto & cvec : params.control_vectors) {
        compat_add_file(buf, cvec.fname);
        compat_add(buf, cvec.strength);
    }
    compat_add(buf, params.control_vector_layer_start);
    compat_add(buf, params.control_vector_layer_end);

    compat_add(buf, (int32_t) params.cache_type_k);
    compat_add(buf, (int32_t) params.cache_type_v);
    compat_add(buf, (uint8_t) params.swa_full);

    compat_add(buf, params.rope_freq_base);
    compat_add(buf, params.rope_freq_scale);
    compat_add(buf, (int32_t) params.rope_scaling_type);
    compat_add(buf, params.yarn_ext_factor);
    compat_add(buf, params.yarn_attn_factor);
    compat_add(buf, params.yarn_beta_fast);
    compat_add(buf, params.yarn_beta_slow);
    compat_add(buf, params.yarn_orig_ctx);

    return hash_sha256_hex(buf.data(), buf.size()).substr(0, 16);
}

//
// server_prompt_cache
//

size_t server_prompt_cache::size() const {
    size_t res = 0;

    for (const auto & state : states) {
        res += state->size();
    }

    return res;
}

size_t server_prompt_cache::n_tokens() const {
    size_t res = 0;

    for (const auto & state : states) {
        res += state->prompt.n_tokens();
    }

    return res;
}

server_prompt_cache_state * server_prompt_cache::alloc(const server_prompt & prompt, size_t state_size_tgt, size_t state_size_dft) {
    // first check if the current state is contained fully in the cache
    for (auto it = states.begin(); it != states.end(); ++it) {
        const int cur_lcp_len = (*it)->prompt.tokens.get_common_prefix(prompt.tokens);

        if (cur_lcp_len == (int) prompt.tokens.size() && are_lora_equal((*it)->prompt.lora, prompt.lora)) {
            SRV_TRC("%s", " - prompt is already in the cache, skipping\n");
            return nullptr;
        }
    }

    // calculate checkpoints size to see if it will fit with the prompt
    size_t checkpoints_size = 0;
    for (const auto & ckpt : prompt.checkpoints) {
        checkpoints_size += ckpt->size();
    }

    const size_t state_size_new = state_size_tgt + state_size_dft + checkpoints_size;

    // skip over-limit entries to avoid disturbing the cache
    if (limit_size > 0 && state_size_new > limit_size) {
        SRV_WRN(" - prompt state size %.3f MiB exceeds cache size limit %.3f MiB, skipping\n",
                state_size_new / (1024.0 * 1024.0), limit_size / (1024.0 * 1024.0));
        return nullptr;
    }

    // remove any cached prompts that are fully contained in the current prompt
    for (auto it = states.begin(); it != states.end();) {
        const int len = (*it)->prompt.tokens.get_common_prefix(prompt.tokens);

        if (len == (int) (*it)->prompt.tokens.size() && are_lora_equal((*it)->prompt.lora, prompt.lora)) {
            SRV_TRC(" - removing obsolete cached prompt with length %d\n", len);

            it = states.erase(it);
        } else {
            ++it;
        }
    }

    if (limit_size > 0) {
        // make room before allocating the new vectors to avoid breaching the limit
        while (!states.empty() && size() + state_size_new > limit_size) {
            SRV_WRN(" - making room for prompt cache entry, removing oldest entry (size = %.3f MiB)\n",
                    (*states.front()).size() / (1024.0 * 1024.0));

            pop_oldest();
        }
    }

    common_state_data state_data_tgt;
    common_state_data state_data_dft;

    // check if we can allocate enough memory for the new state
    try {
        state_data_tgt.resize(state_size_tgt);
        state_data_dft.resize(state_size_dft);
    } catch (const std::bad_alloc & e) {
        SRV_ERR("failed to allocate memory for prompt cache state: %s\n", e.what());

        limit_size = std::max<size_t>(1, 0.4*size());

        SRV_WRN(" - cache size limit reduced to %.3f MiB\n", limit_size / (1024.0 * 1024.0));

        update();

        return nullptr;
    }

    states.push_back(std::make_shared<server_prompt_cache_state>(server_prompt_cache_state{
        /*.prompt =*/ {
            /*.tokens       =*/ prompt.tokens.clone(),
            /*.checkpoints  =*/ prompt.checkpoints,
            /*.lora         =*/ prompt.lora,
            /*.lora_changed =*/ prompt.lora_changed,
        },
        /*.data   =*/ {
            /*.main =*/ std::move(state_data_tgt),
            /*.drft =*/ std::move(state_data_dft),
        },
    }));

    return states.back().get();
}

void server_prompt_cache::save_to_disk(const std::shared_ptr<server_prompt_cache_state> & state) {
    if (!disk || state->prompt.lora_changed) {
        return;
    }

    disk->enqueue(state, disk->pin_core);
}

void server_prompt_cache::flush_to_disk() {
    for (const auto & state : states) {
        save_to_disk(state);
    }
}

bool server_prompt_cache::is_cached(const server_prompt & prompt) const {
    if (prompt.tokens.empty()) {
        return true;
    }

    for (const auto & state : states) {
        // the same containment rule as alloc(): a longer cached prompt covers this one
        if (state->prompt.tokens.get_common_prefix(prompt.tokens) == prompt.tokens.size() &&
                are_lora_equal(state->prompt.lora, prompt.lora)) {
            return true;
        }
    }

    return disk && disk->has(prompt.tokens);
}

size_t server_prompt_cache::get_cached_prefix(const server_prompt & prompt) const {
    if (prompt.tokens.empty()) {
        return 0;
    }

    size_t best = 0;

    for (const auto & state : states) {
        if (!are_lora_equal(state->prompt.lora, prompt.lora)) {
            continue;
        }
        best = std::max(best, (size_t) state->prompt.tokens.get_common_prefix(prompt.tokens));
    }

    if (disk) {
        for (const auto & entry : disk->snapshot()) {
            if (entry.n_tokens == 0) {
                continue;
            }
            best = std::max(best, (size_t) prompt.tokens.get_common_prefix_lower(entry.hashes));
        }
    }

    return best;
}

void server_prompt_cache::pop_oldest() {
    save_to_disk(states.front());
    states.pop_front();
}

void server_prompt_cache::load_from_disk(const server_tokens & tokens_new, const std::vector<common_adapter_lora_info> & lora_new, size_t lcp_slot, int32_t n_ctx_slot) {
    size_t lcp_best = lcp_slot;

    for (const auto & state : states) {
        if (lora_should_clear_cache(state->prompt.lora, lora_new)) {
            continue;
        }
        lcp_best = std::max(lcp_best, (size_t) state->prompt.tokens.get_common_prefix(tokens_new));
    }

    const auto entries = disk->snapshot();

    const server_prompt_cache_disk_entry * entry_best = nullptr;
    const server_prompt_cache_disk_entry * entry_rejected = nullptr;
    size_t lcp_rejected = 0;
    const char * reject_reason = "no prefix match";

    for (const auto & entry : entries) {
        if (entry.n_tokens == 0 || entry.n_tokens > (size_t) n_ctx_slot) {
            if (entry.n_tokens > (size_t) n_ctx_slot && entry.n_tokens > 0) {
                entry_rejected = &entry;
                lcp_rejected = 0;
                reject_reason = "entry exceeds n_ctx";
            }
            continue;
        }

        // only the hashes are known here, so this is a lower bound
        const size_t lcp_cur = tokens_new.get_common_prefix_lower(entry.hashes);

        // don't trash large prompts, but allow reusing a meaningful absolute prefix
        if (lcp_cur < 0.25f * entry.n_tokens && lcp_cur < 256) {
            if (lcp_cur >= lcp_rejected) {
                lcp_rejected = lcp_cur;
                entry_rejected = &entry;
                reject_reason = "lcp too small";
            }
            continue;
        }

        if (lcp_cur > lcp_best) {
            lcp_best = lcp_cur;
            entry_best = &entry;
        }
    }

    if (!entry_best) {
        if (entry_rejected) {
            SRV_DBG("disk prompt cache miss: checked=%zu lcp_best=%zu reason=%s candidate='%s' candidate_tokens=%zu candidate_lcp=%zu new_tokens=%zu\n",
                    entries.size(), lcp_best, reject_reason, entry_rejected->name.c_str(), entry_rejected->n_tokens, lcp_rejected, tokens_new.size());
        } else {
            SRV_DBG("disk prompt cache miss: checked=%zu lcp_best=%zu reason=no usable entries new_tokens=%zu\n",
                    entries.size(), lcp_best, tokens_new.size());
        }
        return;
    }

    SRV_TRC(" - found better prompt on disk with length %zu, lcp >= %zu\n", entry_best->n_tokens, lcp_best);

    server_prompt_cache_state loaded;
    if (disk->load(*entry_best, loaded)) {
        // disk states are only saved for the startup lora config
        loaded.prompt.lora = lora_new;
        SRV_DBG("%s", "__TEST_TAG_PROMPT_CACHE_DISK_LOAD__\n");
        states.push_back(std::make_shared<server_prompt_cache_state>(std::move(loaded)));
    } else {
        SRV_WRN("disk prompt cache load failed: '%s' n_tokens=%zu lcp=%zu\n", entry_best->name.c_str(), entry_best->n_tokens, lcp_best);
    }
}

std::list<std::shared_ptr<server_prompt_cache_state>>::const_iterator server_prompt_cache::find_better(const server_prompt & prompt, const server_tokens & tokens_new, const std::vector<common_adapter_lora_info> & lora_new, int & lcp) const {
    // a prompt made with another lora config is cleared before the new one starts, so it counts as empty
    const bool is_empty = prompt.tokens.empty() || lora_should_clear_cache(prompt.lora, lora_new);

    lcp = is_empty ? 0 : prompt.tokens.get_common_prefix(tokens_new);

    float f_keep_best = is_empty ? -1.0f : float(lcp) / prompt.tokens.size(); // empty slot: any cache entry wins
    float f_sim_best  = float(lcp) / tokens_new.size();

    SRV_TRC(" - looking for better prompt, base f_keep = %.3f, f_sim = %.3f\n", f_keep_best, f_sim_best);

    auto it_best = states.cend();

    // find the most similar cached prompt, that would also preserve the most context
    for (auto it = states.cbegin(); it != states.cend(); ++it) {
        if (lora_should_clear_cache((*it)->prompt.lora, lora_new)) {
            SRV_TRC("   - prompt with length %7zu, skipping, other lora config\n", (*it)->prompt.tokens.size());
            continue;
        }

        // the data cannot be cleared under a background save
        if (disk && disk->busy(**it)) {
            SRV_TRC("   - prompt with length %7zu, skipping, being saved\n", (*it)->prompt.tokens.size());
            continue;
        }

        const int lcp_cur = (*it)->prompt.tokens.get_common_prefix(tokens_new);

        const float f_keep_cur = float(lcp_cur) / (*it)->prompt.tokens.size();
        const float f_sim_cur  = float(lcp_cur) / tokens_new.size();

        SRV_TRC("   - prompt with length %7zu, lcp = %7d, f_keep = %.3f, f_sim = %.3f\n", (*it)->prompt.tokens.size(), lcp_cur, f_keep_cur, f_sim_cur);

        // don't trash large prompts, but allow reusing a meaningful absolute prefix
        if (f_keep_cur < 0.25f && lcp_cur < 256) {
            continue;
        }

        if (f_keep_best < f_keep_cur && f_sim_best < f_sim_cur) {
            f_keep_best = f_keep_cur;
            f_sim_best  = f_sim_cur;
            lcp         = lcp_cur;

            it_best = it;
        }
    }

    if (it_best != states.cend()) {
        SRV_TRC(" - found better prompt with f_keep = %.3f, f_sim = %.3f\n", f_keep_best, f_sim_best);
    }

    return it_best;
}

bool server_prompt_cache::has_better(const server_prompt & prompt, const server_tokens & tokens_new, const std::vector<common_adapter_lora_info> & lora_new) const {
    int lcp = 0;

    return find_better(prompt, tokens_new, lora_new, lcp) != states.cend();
}

bool server_prompt_cache::load(server_prompt & prompt, const server_tokens & tokens_new, const std::vector<common_adapter_lora_info> & lora_new, llama_context * ctx_tgt, llama_context * ctx_dft, int32_t id_slot, int32_t n_ctx_slot) {
    int lcp_best = 0;

    // disk cache is only valid for the startup lora config
    if (disk && !lora_should_clear_cache(lora_init, lora_new)) {
        const int32_t n_ctx_limit = n_ctx_slot > 0 ? n_ctx_slot : INT32_MAX;
        int lcp_disk = prompt.tokens.empty() || lora_should_clear_cache(prompt.lora, lora_new)
            ? 0 : prompt.tokens.get_common_prefix(tokens_new);
        load_from_disk(tokens_new, lora_new, (size_t) lcp_disk, n_ctx_limit);
    }

    const auto it_found = find_better(prompt, tokens_new, lora_new, lcp_best);

    auto it_best = states.end();
    if (it_found != states.cend()) {
        // convert const_iterator to iterator for erase/splice
        it_best = states.begin();
        std::advance(it_best, std::distance(states.cbegin(), it_found));
        it_best = states.erase(it_best, it_best);
    }

    if (it_best != states.end()) {
        const int64_t t_start = ggml_time_us();

        {
            const auto & data = (*it_best)->data.main;

            const size_t size = data.size();
            const size_t n = llama_state_seq_set_data_ext(ctx_tgt, data.data(), size, id_slot, 0);
            if (n != size) {
                SRV_ERR("failed to restore state with size %zu\n", size);

                return false;
            }
        }

        {
            const auto & data = (*it_best)->data.drft;

            if (!data.empty()) {
                GGML_ASSERT(ctx_dft);

                const size_t size = data.size();
                const size_t n = llama_state_seq_set_data_ext(ctx_dft, data.data(), size, id_slot, 0);
                if (n != size) {
                    SRV_WRN("failed to restore state with size %zu\n", size);

                    return false;
                }
            }
        }

        SRV_DBG("%s", "__TEST_TAG_PROMPT_CACHE_RESTORE__\n");

        // the whole cached prompt is a prefix of the new one
        if (lcp_best == (int) (*it_best)->prompt.tokens.size()) {
            SRV_DBG("%s", "__TEST_TAG_PROMPT_CACHE_FULL_RESTORE__\n");
        }

        // keep the state in RAM: the next request often needs the same prompt, and a disk reload costs seconds
        prompt = (*it_best)->prompt.clone();

        // move to the back so the least recently restored prompt is evicted first
        states.splice(states.end(), states, it_best);

        SRV_TRC("slot %d: prompt cache restore took %.2f ms\n", id_slot, (ggml_time_us() - t_start) / 1000.0);
    }

    return true;
}

void server_prompt_cache::update() {
    if (limit_size > 0) {
        while (!states.empty() && size() > limit_size) {
            SRV_WRN(" - cache size limit reached, removing oldest entry (size = %.3f MiB)\n", (*states.front()).size() / (1024.0 * 1024.0));

            pop_oldest();
        }
    }

    // average size per token
    const float size_per_token = std::max<float>(1.0f, float(size()) / (std::max<size_t>(1, n_tokens())));

    // dynamically increase the token limit if it can fit in the memory limit
    const size_t limit_tokens_cur = limit_size > 0 ? std::max<size_t>(limit_tokens, limit_size/size_per_token) : limit_tokens;

    if (limit_tokens > 0) {
        while (!states.empty() && n_tokens() > limit_tokens_cur) {
            SRV_WRN(" - cache token limit (%zu, est: %zu) reached, removing oldest entry (size = %.3f MiB)\n",
                    limit_tokens, limit_tokens_cur, (*states.front()).size() / (1024.0 * 1024.0));

            pop_oldest();
        }
    }

    SRV_TRC(" - cache state: %zu prompts, %.3f MiB (limits: %.3f MiB, %zu tokens, %zu est)\n",
            states.size(), size() / (1024.0 * 1024.0), limit_size / (1024.0 * 1024.0), limit_tokens, limit_tokens_cur);

    for (const auto & state : states) {
        SRV_TRC("   - prompt %p: %7d tokens, checkpoints: %2zu, %9.3f MiB\n",
                (const void *) state.get(), state->prompt.n_tokens(), state->prompt.checkpoints.size(), state->size() / (1024.0 * 1024.0));
    }
}

//
// server_slot_sched
//

static std::vector<const server_slot_info *> get_free_slot_infos(const std::vector<server_slot_info> & slots, size_t n_slots_needed, int exclude_id_slot) {
    std::vector<const server_slot_info *> free_slots;
    for (const auto & slot : slots) {
        if (!slot.is_processing && slot.id != exclude_id_slot) {
            free_slots.push_back(&slot);
        }
    }

    std::stable_sort(free_slots.begin(), free_slots.end(), [](const server_slot_info * a, const server_slot_info * b) {
        if ((a->prompt->n_tokens() == 0) != (b->prompt->n_tokens() == 0)) {
            return a->prompt->n_tokens() == 0;
        }
        return a->t_last_used < b->t_last_used;
    });

    free_slots.resize(std::min(free_slots.size(), n_slots_needed));

    return free_slots;
}

std::vector<int> server_slot_sched::get_free_slots(const std::vector<server_slot_info> & slots, size_t n_slots_needed, int exclude_id_slot) const {
    std::vector<int> res;
    for (const auto * slot : get_free_slot_infos(slots, n_slots_needed, exclude_id_slot)) {
        res.push_back(slot->id);
    }

    return res;
}

server_slot_plan server_slot_sched::select(const std::vector<server_slot_info> & slots, const server_task & task, const std::vector<common_adapter_lora_info> & lora, const server_prompt_cache * prompt_cache) const {
    const server_slot_info * ret = nullptr;

    bool update_cache = false;

    // if a specific slot is requested, use it (still goes through cache update logic below)
    if (task.id_slot != -1 && !slots.empty()) {
        // note: allow id_slot to be out of bounds (wrap around)
        const int id_slot = task.id_slot % slots.size();

        for (const auto & slot : slots) {
            if (slot.id == id_slot) {
                ret = &slot;
                break;
            }
        }

        if (ret) {
            // a busy slot is returned untouched, the caller defers the task
            if (ret->is_processing) {
                return { ret->id, false };
            }

            SLT_INF(*ret, "selected slot by id (%d)\n", task.id_slot);
        }
    }

    // find the slot that has at least n% prompt similarity
    if (slot_prompt_similarity != 0.0f) {
        float f_sim_best = 0;

        for (const auto & slot : slots) {
            if (task.id_slot != -1 && slot.id != task.id_slot) {
                continue;
            }

            // skip the slot if it is not available
            if (slot.is_processing) {
                SLT_TRC(slot, " - skipping, is_processing = %d\n", slot.is_processing);
                continue;
            }

            const auto & tokens = slot.prompt->tokens;

            // skip the slot if it does not contains cached tokens
            if (tokens.empty()) {
                SLT_TRC(slot, "%s", " - skipping, slot is empty\n");
                continue;
            }

            // the task would clear this slot, its state cannot be reused with another lora config
            if (lora_should_clear_cache(slot.prompt->lora, lora)) {
                SLT_TRC(slot, "%s", " - skipping, other lora config\n");
                continue;
            }

            // fraction of the Longest Common Prefix length with respect to the input prompt length
            const size_t lcp_len = tokens.get_common_prefix(task.tokens);
            const float f_sim_cur = float(lcp_len) / task.tokens.size();

            SLT_TRC(slot, " - checking sim = %.3f (%zu/%zu) > %.3f\n", f_sim_cur, lcp_len, task.tokens.size(), slot_prompt_similarity);

            // select the current slot if the criteria match
            if (f_sim_cur > f_sim_best && f_sim_cur > slot_prompt_similarity) {
                f_sim_best = f_sim_cur;

                ret = &slot;
            }
        }

        if (ret != nullptr) {
            // empty slot: nothing to keep, avoid 0/0
            const float f_keep = ret->prompt->tokens.size() > 0 ?
                (f_sim_best*task.tokens.size()) / ret->prompt->tokens.size() : 0.0f;

            if (task.id_slot == -1) {
                SLT_INF(*ret, "selected slot by LCP similarity, f_sim_best = %.3f (> %.3f thold), f_keep = %.3f\n",
                        f_sim_best, slot_prompt_similarity, f_keep);
            }

            // if we are about to lose a large portion of the existing context - save it in the prompt cache
            if (f_keep < 0.5f) {
                update_cache = true;
            }
        }
    }

    // find an empty slot, or the one that has been least recently used
    if (ret == nullptr) {
        const auto free_slots = get_free_slot_infos(slots, 1, -1);

        if (!free_slots.empty()) {
            ret = free_slots[0];

            SLT_INF(*ret, "selected slot by LRU, n_tokens = %d, t_last = %" PRId64 "\n", ret->prompt->n_tokens(), ret->t_last_used);

            update_cache = true;
        }
    }

    if (ret == nullptr) {
        return {};
    }

    // cache prompts only for the tasks that reuse them
    if (prompt_cache == nullptr || task.is_stateless()) {
        return { ret->id, false };
    }

    // the checks above only tell if the outgoing state is worth saving
    // the cache can still hold a better start for this task, so ask it
    update_cache = update_cache || prompt_cache->has_better(*ret->prompt, task.tokens, lora);

    return { ret->id, update_cache };
}
