# Actual Training Epochs in Reports Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add the actual completed training epoch count to generated report files, especially when early stopping ends training before configured `epochs`.

**Architecture:** Store `actual_epoch_nums` as runtime experiment state in the existing `ConfigManager`, update it from `src/trainer.py` after each completed epoch, and have `src/metrics.py` print it after `Report time` and before OA/AA/Kappa. This keeps output-path naming based on configured epochs unchanged while making report contents reflect the completed training run.

**Tech Stack:** Python, unittest, existing `src.config` global config manager, existing trainer/report pipeline.

---

## File Structure

- Modify `HSI-X-classify-backbone/src/config.py`: add `actual_epoch_nums` to `ExperimentConfig`, expose it via `_parameter_dict`, and support `set_value('actual_epoch_nums', value)`.
- Modify `HSI-X-classify-backbone/src/trainer.py`: initialize actual epoch count before training and update it after each completed epoch.
- Modify `HSI-X-classify-backbone/src/metrics.py`: write `Actual training epochs: N` after `Report time` when `actual_epoch_nums` is available.
- Modify `HSI-X-classify-backbone/test_early_stopping_config.py`: cover config storage for `actual_epoch_nums`.
- Modify `HSI-X-classify-backbone/test_email_notifier.py`: cover report summary including the new line before metrics.

### Task 1: Add runtime config field

**Files:**
- Modify: `HSI-X-classify-backbone/src/config.py:1-240`
- Test: `HSI-X-classify-backbone/test_early_stopping_config.py`

- [ ] **Step 1: Write the failing config test**

Add this test method to `EarlyStoppingConfigTests` in `HSI-X-classify-backbone/test_early_stopping_config.py`:

```python
    def test_actual_epoch_nums_can_be_stored_as_runtime_state(self):
        manager = ConfigManager(ExperimentConfig())

        self.assertIsNone(manager.get_value('actual_epoch_nums'))

        manager.set_value('actual_epoch_nums', 32)

        self.assertEqual(manager.get_value('actual_epoch_nums'), 32)
        self.assertEqual(manager.config.actual_epoch_nums, 32)
```

- [ ] **Step 2: Run test to verify it fails**

Run from `HSI-X-classify-backbone`:

```bash
python -m unittest test_early_stopping_config.EarlyStoppingConfigTests.test_actual_epoch_nums_can_be_stored_as_runtime_state
```

Expected: FAIL because `ExperimentConfig` has no `actual_epoch_nums` field and/or `ConfigManager.set_value()` does not sync it.

- [ ] **Step 3: Add `actual_epoch_nums` to config**

In `HSI-X-classify-backbone/src/config.py`, add this dataclass field near the training parameters in `ExperimentConfig`:

```python
    actual_epoch_nums: Optional[int] = None
```

Then add this entry to `_create_parameter_dict()` near `'epoch_nums': self.config.epochs`:

```python
            'actual_epoch_nums': self.config.actual_epoch_nums,
```

Then add this branch to `set_value()` immediately after the `epoch_nums` branch:

```python
        elif key == 'actual_epoch_nums':
            self.config.actual_epoch_nums = value
```

- [ ] **Step 4: Run config test to verify it passes**

Run from `HSI-X-classify-backbone`:

```bash
python -m unittest test_early_stopping_config.EarlyStoppingConfigTests.test_actual_epoch_nums_can_be_stored_as_runtime_state
```

Expected: PASS.

- [ ] **Step 5: Run existing early stopping config tests**

Run from `HSI-X-classify-backbone`:

```bash
python -m unittest test_early_stopping_config
```

Expected: all tests pass.

### Task 2: Update actual epoch count during training

**Files:**
- Modify: `HSI-X-classify-backbone/src/trainer.py:69-185`

- [ ] **Step 1: Initialize runtime state before training starts**

In `train()` in `HSI-X-classify-backbone/src/trainer.py`, immediately before `for epoch in range(epochs):`, add:

```python
    config.set_value('actual_epoch_nums', 0)
```

- [ ] **Step 2: Update runtime state after each completed epoch**

In `train()` in `HSI-X-classify-backbone/src/trainer.py`, inside the `if epoch % 1 == 0:` block, after `acc1 = accuracy_score(gty, y_pred_test)` and before `if acc1 > max_acc:`, add:

```python
            config.set_value('actual_epoch_nums', epoch + 1)
```

This makes an early stop at epoch 32 leave `actual_epoch_nums == 32`.

- [ ] **Step 3: Run early stopping config tests**

Run from `HSI-X-classify-backbone`:

```bash
python -m unittest test_early_stopping_config
```

Expected: all tests pass.

### Task 3: Write actual training epochs in report output

**Files:**
- Modify: `HSI-X-classify-backbone/src/metrics.py:61-76`
- Test: `HSI-X-classify-backbone/test_email_notifier.py`

- [ ] **Step 1: Write the failing report-summary test**

Add this test method to `EmailNotifierTests` in `HSI-X-classify-backbone/test_email_notifier.py`:

```python
    def test_extract_report_summary_includes_actual_training_epochs_line(self):
        report_text = """-----------------------taskInfo-----------------------\nlr:\t0.0001\nepoch_nums:\t100\n------------------------------------------------------\n\nReport time: 2026-05-21 10:00:00\n\nActual training epochs: 32\n75.17 Overall accuracy (%)\n70.00 Average accuracy (%)\n65.00 Kappa accuracy (%)\n\nclassification details\n"""
        report_path = PROJECT_ROOT / "tmp_test_report.txt"
        report_path.write_text(report_text, encoding="utf-8")
        self.addCleanup(lambda: report_path.unlink(missing_ok=True))

        summary = extract_report_summary(report_path)

        self.assertIn("Actual training epochs: 32", summary)
        self.assertIn("75.17 Overall accuracy (%)", summary)
```

- [ ] **Step 2: Run test to establish current summary behavior**

Run from `HSI-X-classify-backbone`:

```bash
python -m unittest test_email_notifier.EmailNotifierTests.test_extract_report_summary_includes_actual_training_epochs_line
```

Expected: PASS if the summary already includes all lines through Kappa. If it fails, inspect `src/email_notifier.py` and adjust summary extraction only enough to include the line between `Report time` and OA/AA/Kappa.

- [ ] **Step 3: Write actual epoch line in `metrics.py`**

In `getReport()` in `HSI-X-classify-backbone/src/metrics.py`, after these lines:

```python
        report.write('{}'.format(current_time_log))
        report.write('\n')
```

add:

```python
        actual_epoch_nums = config.get_value('actual_epoch_nums')
        if actual_epoch_nums is not None:
            report.write('Actual training epochs: {}'.format(actual_epoch_nums))
            report.write('\n')
```

The resulting report order must be:

```text
Report time: 2026-05-21 10:00:00

Actual training epochs: 32
75.17 Overall accuracy (%)
```

- [ ] **Step 4: Run email notifier tests**

Run from `HSI-X-classify-backbone`:

```bash
python -m unittest test_email_notifier
```

Expected: all tests pass.

### Task 4: Verify the whole focused test set

**Files:**
- Test: `HSI-X-classify-backbone/test_early_stopping_config.py`
- Test: `HSI-X-classify-backbone/test_email_notifier.py`

- [ ] **Step 1: Run focused tests together**

Run from `HSI-X-classify-backbone`:

```bash
python -m unittest test_early_stopping_config test_email_notifier
```

Expected: all tests pass.

- [ ] **Step 2: Smoke-train if local data and dependencies are available**

Run from `HSI-X-classify-backbone`:

```bash
python scripts/strain.py
```

Expected: training completes for 1 epoch and any generated report after evaluation contains `Actual training epochs: 1` after `Report time`. If local data or dependencies are unavailable, record the exact missing dependency/data error and do not claim smoke verification passed.

## Self-Review

- Spec coverage: the plan stores actual completed epochs, preserves configured epoch path naming, handles early stopping by updating after each completed epoch, and writes the value after `Report time` before metrics.
- Placeholder scan: no TBD/TODO/placeholders remain; every edit step gives concrete code and commands.
- Type consistency: the runtime key is consistently named `actual_epoch_nums`, with type `Optional[int]`, and report text is consistently `Actual training epochs: N`.
