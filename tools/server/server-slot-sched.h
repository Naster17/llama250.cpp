#pragma once

#include "server-common.h"
#include "server-task.h"

#include <atomic>
#include <condition_variable>
#include <deque>
#include <filesystem>
#include <list>
#include <memory>
#include <mutex>
#include <set>
#include <string>
#include <thread>
#include <vector>

// which prompt goes to which slot: slot prompt state, prompt cache and slot selection

// what a cached prompt can give to a new prompt
struct server_prompt_reuse {
    int32_t   n_past   = 0; // tokens to keep
    llama_pos pos_next = 0;

    // restore this checkpoint first, nullptr when the live state is used as it is
    std::shared_ptr<const common_prompt_checkpoint> ckpt;
};

struct server_prompt {
    server_tokens tokens;

    // a checkpoint does not change after it is created, so the prompt cache shares it with the slot instead of a copy
    std::list<std::shared_ptr<const common_prompt_checkpoint>> checkpoints;

    // the lora config that the tokens were processed with, it follows the state to the prompt cache and back
    // TODO: slot files do not store it
    std::vector<common_adapter_lora_info> lora;

    // some tokens were processed with a lora config other than the one at startup
    // TODO: such a prompt is not saved to disk, because the lora config is not a part of the prompt hash
    bool lora_changed = false;

    void clear() {
        tokens.clear();
        checkpoints.clear();
        lora.clear();
        lora_changed = false;
    }

    int n_tokens() const {
        return tokens.size();
    }

    server_prompt clone() const {
        return server_prompt {
            tokens.clone(),
            checkpoints,
            lora,
            lora_changed,
        };
    }

    // n_prefix: tokens shared with the new prompt, must be > 0
    // has_new_tokens: the new prompt is longer than n_prefix
    // n_swa: 0 if a state must end right before the next token (no SWA, or recurrent)
    // pos_min: min position of the live state
    // n_past == 0 without a checkpoint means that nothing can be reused
    server_prompt_reuse get_reuse(int32_t n_prefix, bool has_new_tokens, int32_t n_swa, llama_pos pos_min) const;
};

struct server_prompt_data {
    common_state_data main;
    common_state_data drft;

    size_t size() const {
        return main.size() + drft.size();
    }
};

struct server_prompt_cache_state {
    server_prompt prompt;
    server_prompt_data data;

    size_t size() const {
        size_t res = data.size();

        for (const auto & ckpt : prompt.checkpoints) {
            res += ckpt->size();
        }

        return res;
    }
};

// prompt cache on disk, one file per prompt in a flat directory: {compat}-{prompt hash}.kvc
// file: header | hash chain | tokens | state | draft state | checkpoints

struct server_prompt_cache_disk_entry {
    std::string name;

    size_t size = 0; // in bytes
    std::filesystem::file_time_type t_used;

    // the fields below are not set for the files of another compat id, such files only count in the size limit
    size_t n_tokens = 0;

    server_tokens::prefix_hash hash = {}; // of the whole prompt
    std::vector<server_tokens::prefix_hash> hashes;

    bool pinned = false; // core/base prompt: never evicted by size limit or replaced by a longer prompt
};

struct server_prompt_cache_disk {
    // compat: 16 hex chars, changes when the saved states can no longer be used (other model, mmproj, etc.)
    // limit_size: in bytes, 0 = no limit
    // n_threads: number of background writers, normally -t, so saves of different prompts run in parallel on the allowed CPU cores
    server_prompt_cache_disk(const std::string & dir, const std::string & compat, size_t limit_size, bool has_mtmd, int32_t n_threads = 4);

    // does not wait for the disk: a save in progress stops at the next chunk, the queued ones are dropped
    ~server_prompt_cache_disk();

    // pass entry.hashes to server_tokens::get_common_prefix_lower() to find a matching prompt without reading its file
    // note: not thread-safe, use snapshot() while the background writers run
    const std::list<server_prompt_cache_disk_entry> & get_entries() const { return entries; }

    // a copy of the entries for use while the background writers run
    std::vector<server_prompt_cache_disk_entry> snapshot() const;

    size_t size() const;

    // removes the saved prompts that are a prefix of this one, then the oldest files if over the limit
    // pinned entries are never removed by size limit and never replaced by a longer prompt prefix
    // note: synchronous, the caller does the disk write
    bool save(const server_prompt_cache_state & state, bool pinned = false);

    // save in the background; a state that is already queued, written or on disk is skipped
    void enqueue(std::shared_ptr<const server_prompt_cache_state> state, bool pinned = false);

    // wait until no background save is queued or running
    void wait();

    // whether this state is queued for a background save or is being written
    bool busy(const server_prompt_cache_state & state) const;

    // whether this prompt is already on disk or waiting in the queue
    bool has(const server_tokens & tokens) const;

    // a file that cannot be read is removed
    bool load(const server_prompt_cache_disk_entry & entry, server_prompt_cache_state & state);

    // mark an entry as pinned (core/base prompt). Rewrites the header flag on disk.
    void pin(const server_prompt_cache_disk_entry & entry);

    bool pin_core = false; // pin the first saved prompt prefix as a core/base prompt

    // warm the MRU disk entries into the OS page cache in the background, so the first load does not stall on disk I/O
    void start_warmup();

private:
    using entry_list = std::list<server_prompt_cache_disk_entry>;

    struct job {
        std::shared_ptr<const server_prompt_cache_state> state;
        std::string name;
        bool pinned = false;
    };

    // cancelable: the background workers stop early when the server shuts down
    bool save_impl(const server_prompt_cache_state & state, bool pinned, bool cancelable);

    void scan();

    entry_list::iterator remove_unlocked(entry_list::iterator it);
    void touch_unlocked(entry_list::iterator it);
    void pin_unlocked(entry_list::iterator it);
    size_t size_unlocked() const;

    void worker_loop();

    std::filesystem::path dir;
    std::string compat;
    size_t limit_size;
    bool has_mtmd;

    // oldest first
    entry_list entries;

    mutable std::mutex mutex;
    std::condition_variable cv_work;
    std::condition_variable cv_done;
    std::deque<job> queue;
    std::set<std::string> queued;  // names waiting in the queue
    std::set<std::string> writing; // names being written, protected by mutex
    size_t writing_size = 0;       // bytes reserved by the writes in progress, so parallel writers respect limit_size
    std::vector<std::thread> workers;
    std::thread warmup_thread;
    size_t n_workers;
    size_t n_active = 0;           // writes in progress
    std::atomic<bool> stopping{false};
};

// id of everything that makes a saved state unusable when it changes
std::string server_prompt_cache_disk_compat(const common_params & params);

struct server_prompt_cache {
    server_prompt_cache(int32_t limit_size_mib, size_t limit_tokens) {
        this->limit_size   = 1024ull*1024ull*(limit_size_mib < 0 ? 0 : limit_size_mib);
        this->limit_tokens = limit_tokens;
    }

    std::list<std::shared_ptr<server_prompt_cache_state>> states;

    // optional, the states that are dropped from the RAM are saved there
    std::unique_ptr<server_prompt_cache_disk> disk;

    // the lora config the disk cache is valid for (startup config)
    std::vector<common_adapter_lora_info> lora_init;

    // in bytes, 0 = no limit
    size_t limit_size = 0;

    // in tokens, 0 = no limit
    size_t limit_tokens = 0;

    size_t size() const;

    size_t n_tokens() const;

    server_prompt_cache_state * alloc(const server_prompt & prompt, size_t state_size_main, size_t state_size_drft);

    // lora_new: the lora config of the new prompt, the states made with another config are skipped
    // n_ctx_slot: longer prompts are not loaded from the disk
    bool load(server_prompt & prompt, const server_tokens & tokens_new, const std::vector<common_adapter_lora_info> & lora_new, llama_context * ctx_tgt, llama_context * ctx_dft, int32_t id_slot, int32_t n_ctx_slot);

    void update();

    // saves all the states to disk in the background, they stay in the RAM
    void flush_to_disk();

    // whether the prompt is already in the RAM cache or on disk, so it does not need another GPU readback
    bool is_cached(const server_prompt & prompt) const;

    // longest cached prefix for this prompt in RAM or on disk (lower bound for disk entries)
    size_t get_cached_prefix(const server_prompt & prompt) const;

private:
    void save_to_disk(const std::shared_ptr<server_prompt_cache_state> & state);

    void pop_oldest();

    // moves the best match from the disk to the RAM if it beats the slot and the states in RAM
    void load_from_disk(const server_tokens & tokens_new, const std::vector<common_adapter_lora_info> & lora_new, size_t lcp_slot, int32_t n_ctx_slot);
};

// what the scheduler needs to know about a slot
struct server_slot_info {
    int id = -1;
    const server_task * task = nullptr; // only for logging

    bool is_processing = false;
    int64_t t_last_used = -1;

    const server_prompt * prompt = nullptr;
};

struct server_slot_plan {
    // -1 if no slot is available; a busy slot if the task asks for it by id, the caller defers the task
    int id_slot = -1;

    // save the slot state to the prompt cache, then load the best match for the task
    bool update_cache = false;
};

// decides only, the caller applies the plan
struct server_slot_sched {
    float slot_prompt_similarity = 0.0f;

    // idle slots that lose the least by being reused: empty slots first, then the least recently used ones
    std::vector<int> get_free_slots(const std::vector<server_slot_info> & slots, size_t n_slots_needed, int exclude_id_slot) const;

    // lora: the lora config of the task
    server_slot_plan select(const std::vector<server_slot_info> & slots, const server_task & task, const std::vector<common_adapter_lora_info> & lora) const;
};
