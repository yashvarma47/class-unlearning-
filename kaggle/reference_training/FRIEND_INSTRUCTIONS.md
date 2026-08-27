# Instructions — training one reference model on Kaggle

Thank you for helping. This takes about an hour of GPU time per class and almost no
effort from you, **as long as you change nothing**.

> **Do not start yet.** Wait until Yash tells you the ship trial has passed. If you have
> not had that message, stop here.

---

## The rules

These are strict because every model in this experiment has to be trained the *same way*.
One changed number and the model cannot be compared with the others, and the run is
wasted.

* **Do not change the code.** Not a line.
* **Do not change the classes.** Run only the assignment file Yash gave you.
* **Do not change the hyperparameters.** Not the epochs, not the learning rate, not the
  batch size, not the seed.
* **Do not train frog (class 6).** It is already finished. The script will refuse anyway.
* **Do not run the MED-US search.** Not part of this.
* **Do not run any baselines.** Not part of this.
* **Run the notebook as it is**, top to bottom.
* **Download the output zip.**
* **Send the zip back to Yash**, unrenamed and unopened.

If something looks wrong, **stop and message Yash**. Do not try to fix it. A failed run
costs an hour; a silently wrong run costs the experiment.

---

## Setup

1. Open Kaggle → **Code** → **New Notebook**, then **File → Import Notebook** and upload
   `train_references_kaggle.ipynb`.
2. In the right-hand panel:
   * **Accelerator:** GPU (T4 or P100)
   * **Internet:** On
3. **Add Input** → search Kaggle Datasets for **`medus-class-code`** (Yash will share the
   link) → Add.

There is no second dataset. The models are trained from scratch.

---

## Running

### Step 1 — set your assignment file

The very first code cell has one line to edit:

```python
ASSIGNMENT = "trial_ship.yaml"
```

Change it to the file Yash gave you:

| you | line to use |
|---|---|
| Pragati | `ASSIGNMENT = "classes_pragati.yaml"` |
| Aditya | `ASSIGNMENT = "classes_aditya.yaml"` |

**That is the only edit you make anywhere in the notebook.**

### Step 2 — Run All

Then read what cell 4 prints. It trains nothing; it just shows you what is about to
happen:

```
assigned person      <your name>
forget class id      <a number>
forget class name    <a word>
D_f_train            5000
D_r_train            45000
D_f_test             1000
D_r_test             9000
seed                 42
epochs               200
```

**Check three things:**

1. The name is yours.
2. The classes are the ones you were given.
3. The four sizes are exactly **5000 / 45000 / 1000 / 9000**.

If any of those is wrong, **stop and message Yash.** Do not run the next cell.

### Step 3 — wait

The training cell takes roughly **45–75 minutes per class**. Leave the tab open; Kaggle
stops a session that looks idle. If your file lists three classes, expect 2–4 hours total.

### Step 4 — download

When the last cell finishes it prints a file name like:

```
DOWNLOAD THIS FILE: reference_outputs_pragati.zip
```

Open the **Output** panel on the right, download that zip, and send it to Yash.

Do not rename it. Do not unzip it. Do not send anything else — the zip already contains
everything needed.

---

## If something goes wrong

| what you see | what to do |
|---|---|
| `no GPU` | Set Accelerator to GPU in the settings panel and rerun |
| `code dataset not found` | Add the `medus-class-code` input dataset |
| `assignment file ... not found` | You typed the file name wrong in cell 0 |
| `ABORT: split sizes are wrong` | Stop. Send Yash the whole output. |
| `class 6 (frog) is already finished` | You picked the wrong assignment file |
| Session died partway | Rerun from the top. Nothing is lost; it just starts over. |

Any error at all: copy the **full** output of the failing cell and send it to Yash. Please
do not edit anything to make an error go away.
