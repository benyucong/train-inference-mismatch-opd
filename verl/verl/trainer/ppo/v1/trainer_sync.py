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
import logging
import os

from verl.trainer.ppo.v1.trainer_base import PPOTrainer, register_trainer
from verl.utils.debug import marked_timer

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "INFO"))


@register_trainer("sync")
class PPOTrainerSync(PPOTrainer):
    """Synchronous PPO trainer
    1. Trainer and rollout are colocated
    2. Partial rollout is disabled
    """

    def on_init_end(self):
        # update weights after loading checkpoint
        self.checkpoint_manager.update_weights(self.global_steps)

    def on_validate_end(self):
        # SPECOPD: validation traffic must not leak into the next training step's telemetry (opt-in).
        if os.environ.get("SPECOPD_TELEMETRY", "0") not in ("0", "", "false", "False"):
            from verl.utils.specopd_telemetry import discard_validation

            discard_validation(self.llm_server_manager)

    def on_step_end(self):
        with marked_timer("update_weights", self.timing_raw, color="red"):
            # wake up all replicas to update weights
            self.checkpoint_manager.update_weights(self.global_steps)

    def on_sample_end(self):
        # SPECOPD: pull the rollout engines' spec-decode telemetry BEFORE they sleep (opt-in SPECOPD_TELEMETRY=1;
        # joined with the per-request records in trainer_base._step_once). This runs inside the "gen" timer, so it
        # gets its own: compare arms on timing_s/gen - timing_s/specopd_pull.
        if os.environ.get("SPECOPD_TELEMETRY", "0") not in ("0", "", "false", "False"):
            from verl.utils.specopd_telemetry import pull_engines

            with marked_timer("specopd_pull", self.timing_raw, color="green"):
                self._specopd_pulled = pull_engines(self.llm_server_manager.server_handles)
        # sleep all replicas to discard weights and kv cache
        self.checkpoint_manager.sleep_replicas()
        if self.curr_step_profile:
            self._stop_rollout_profiling()
