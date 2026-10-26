# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from verl.trainer.ppo.v1.trainer_base import PPOTrainer, register_trainer
from verl.utils.debug import marked_timer
from verl.workers.rollout.llm_server import FullyAsyncLLMServerClient


@register_trainer("colocate_async")
class PPOTrainerColocateAsync(PPOTrainer):
    """Asynchronous PPO trainer
    1. Trainer and rollout are colocated.
    2. Partial rollout is enabled.
    """

    def get_llm_client(self):
        """Get the LLM server client for rollout generation."""
        return self.llm_server_manager.get_client(client_cls=FullyAsyncLLMServerClient)

    def on_init_end(self):
        # update weights after loading checkpoint
        self.checkpoint_manager.update_weights(self.global_steps)

    def on_train_begin(self):
        self._add_async_warmup_batches(self.config.trainer.v1.colocate_async.num_warmup_batches)

    def on_step_end(self):
        with marked_timer("update_weights", self.timing_raw, color="red"):
            # wake up all replicas to update weights
            self.checkpoint_manager.update_weights(self.global_steps)
            # resume generation
            # SPECOPD: own (nested) timer, so update_weights - resume_generation is the weight sync alone
            with marked_timer("resume_generation", self.timing_raw, color="red"):
                self.checkpoint_manager.resume_generation_replicas()

    def on_sample_end(self):
        # SPECOPD: this step's sampler wait accounting (training/async/*, ReplayBufferAsync), merged into
        # the step's metrics through the per-step sync-metrics stash that fit() consumes after on_step_end.
        wait_stats = getattr(self.replay_buffer, "last_wait_stats", {}).pop("train", None)
        if wait_stats:
            self._pending_sync_metrics = {**(getattr(self, "_pending_sync_metrics", None) or {}), **wait_stats}
        # SPECOPD: abort and sleep run inside the "gen" timer; own (nested) timers, so in this mode
        # timing_s/gen - abort_replicas - sleep_replicas = the trainer waiting for finished groups (not generation).
        # abort all unfinished requests and pause generation
        with marked_timer("abort_replicas", self.timing_raw, color="red"):
            self.checkpoint_manager.abort_replicas()
        # sleep all replicas to discard weights and kv cache
        with marked_timer("sleep_replicas", self.timing_raw, color="red"):
            self.checkpoint_manager.sleep_replicas()
        if self.curr_step_profile:
            self._stop_rollout_profiling()
