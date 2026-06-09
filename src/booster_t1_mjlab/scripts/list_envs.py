"""Script to list mjlab environments."""

import tyro
from prettytable import PrettyTable

import mjlab
import booster_t1_mjlab.tasks  # noqa: F401
import mjlab.tasks  # noqa: F401
from booster_t1_mjlab.tasks.registry import list_tasks
from mjlab.tasks.registry import list_tasks as list_mjlab_tasks, load_rl_cfg as mjlab_load_rl_cfg


def list_environments(keyword: str | None = None):
  """List all registered environments.

  Args:
    keyword: Optional filter to only show environments containing this keyword.
  """
  from beyondAMP.mjlab.rsl_rl import AMPRunnerCfg

  def matches(task_id: str) -> bool:
    return not keyword or keyword.lower() in task_id.lower()

  # --- PPO tasks (booster registry) ---
  ppo_table = PrettyTable(["#", "Task ID"])
  ppo_table.title = "PPO Tasks  (train: booster_t1_train)"
  ppo_table.align["Task ID"] = "l"
  for i, task_id in enumerate((t for t in list_tasks() if matches(t)), start=1):
    ppo_table.add_row([i, task_id])

  # --- AMP tasks (mjlab registry) ---
  amp_tasks = [
    t for t in list_mjlab_tasks()
    if isinstance(mjlab_load_rl_cfg(t), AMPRunnerCfg) and matches(t)
  ]
  amp_table = PrettyTable(["#", "Task ID"])
  amp_table.title = "AMP Tasks  (train: booster_t1_train_beyondamp)"
  amp_table.align["Task ID"] = "l"
  for i, task_id in enumerate(amp_tasks, start=1):
    amp_table.add_row([i, task_id])

  print(ppo_table)
  print()
  print(amp_table)


def main():
  tyro.cli(list_environments, config=mjlab.TYRO_FLAGS)


if __name__ == "__main__":
  main()
