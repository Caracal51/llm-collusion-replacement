# Replacing the Agent Doesn't Reset the Price

Code and data for the paper *Replacing the Agent Doesn't Reset the Price* (Apart Research AI
Collusion Research Sprint, 23–25 October 2026, Track 1, question 1.2).

**In one paragraph.** Two Qwen2.5-14B sellers play a repeated logit-demand duopoly (the Calvano et al.
2020 / Fish et al. 2024 setup). Games that reach profitable coordination are checkpointed and branched
into arms that replace one or both agents, restrict the price history they see, or wipe it. Replacing
the agent (even with Microsoft's Phi-4) leaves coordination intact; only wiping the shared price record
ends profitable coordination, and even then prices do not fall. Follow-up transplant and
fabricated-record experiments show the record sets the price, and the result replicates with Phi-4.

**Disclosure.** All code in this repository was written during the sprint (23–25 Oct 2026). The market
model follows Calvano et al. (2020) and Fish et al. (2024); the prompt follows Fish et al.'s P1 prefix,
in our own wording.

## Check every number in the paper (no GPU, about a minute)

```bash
mkdir check && cd check
unzip ../data/results.zip -d r2        # Qwen runs (Saturday): main, b0, pilots, seeded, transplant
unzip ../data/results_phi.zip -d r3    # Phi-4 replication (Sunday)
python3 -I ../verify_paper.py r2/runs r3/runs
```

`verify_paper.py` reads only the raw game files (it does not use any of our analysis code or summary
files), recomputes each number the paper reports, prints one line per claim (paper value next to the
recomputed value) and writes `verify.json`. Or upload the two zips to Kaggle and run
`notebooks/verify_numbers_kaggle.ipynb`, which ends with `ALL MATCH` if every number agrees.

## What is in the repo

| Path | What it is |
| --- | --- |
| `market.py` | Demand, profits, Nash and monopoly benchmarks, profit and price indexes |
| `agents.py` | Prompts, response parsing, model clients (OpenAI-compatible and a mock) |
| `experiment.py` | `ping`, `base`, `branch`, `smoke` commands; checkpointing and arms |
| `analyze.py` | Metrics, paired bootstrap CIs, sign-flip tests, Holm correction, `summary.md` |
| `extensions.py` | Follow-ups: fabricated records (`seed`), real-record `transplant`, `deviate` (punishment test) |
| `extra_analysis.py` | Price-index robustness, anchoring, consumer harm, newcomers' notes |
| `paper_figs.py` | The paper's figures |
| `modal_vllm.py` | Serves an open-weight model on a Modal GPU with vLLM 0.11.0 |
| `verify_paper.py` | Independent re-computation of every number in the paper |
| `notebooks/collusion_kaggle.ipynb` | Saturday: main experiment, run from Kaggle with Modal GPUs |
| `notebooks/phi4_replication.ipynb` | Sunday: Phi-4 replication |
| `notebooks/verify_numbers_kaggle.ipynb` | Runs `verify_paper.py` on Kaggle |
| `data/results.zip` | Every raw game, checkpoint, branch and full prompt/response log (Qwen runs) |
| `data/results_phi.zip` | The same for the Phi-4 replication |

Models: M = `Qwen/Qwen2.5-14B-Instruct`; P = `microsoft/phi-4` (the plan named Llama-3.1-8B; we
switched to Phi-4 on Saturday before running arms A2 and A4). Arm A3 (`Qwen2.5-7B`) was not run.

## Arms

Seller A is the incumbent, seller B is replaced. LLM calls are stateless, so an agent is exactly its
model weights plus its notes.

| Arm | After the swap |
| --- | --- |
| A0 | No change (control) |
| A1 | Seller B: fresh Qwen, notes wiped |
| A2 | Seller B: Phi-4, keeps B's notes (handover) |
| A4 | Seller B: Phi-4, notes wiped |
| A5 | Both sellers' notes wiped, price record kept |
| A6 | Both sellers' notes and the price record wiped |
| A7 | Agents see only the last 3 rounds of the record |
| A8 | Rival prices hidden from the record |


## Re-running the experiments (needs Modal GPUs)

### From Kaggle (no laptop)

Upload `notebooks/collusion_kaggle.ipynb` to Kaggle (Create → New Notebook → File → Import Notebook) and
follow the instructions at its top. It contains all the code, deploys the model to Modal, and runs
every step below. The rest of this README is for running from a terminal instead.

### 1. Set up (laptop, 5 minutes)

```bash
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python experiment.py smoke                             # offline test, no GPU, no keys
```

The smoke test uses scripted fake agents. Its numbers mean nothing; it only proves the pipeline
works. Output lands in `runs/smoke/`.

### 2. Start a model on Modal

```bash
modal setup                                             # log in once
python -c "import secrets; print(secrets.token_urlsafe(32))"   # make a random key
modal secret create vllm-api-key VLLM_API_KEY=<that key>
modal secret create huggingface HF_TOKEN=<your HF token>   # free HF account; needed for gated models
cp .env.example .env                                    # put the same key in .env

MODEL=Qwen/Qwen2.5-14B-Instruct APP_NAME=collusion-m GPU=L40S modal deploy modal_vllm.py
```

Copy `models.example.json` to `models.json` and paste the URL Modal printed, adding `/v1`.
Then check it answers (the first call waits several minutes while the GPU starts and the weights
download):

```bash
python experiment.py ping --keys M
```

For arms A2 and A4, deploy model P (`microsoft/phi-4`) the same way with `APP_NAME=collusion-p`.

### 3. Pilot: does this model collude? (Friday)

```bash
python experiment.py base --out runs/pilot --n 3 --rounds 100
python analyze.py --out runs/pilot --base-only
```

Look at `runs/pilot/figs/fig0_base_runs.png`. You need runs whose Δ stays at or above 0.5.
If none do, try another model. Set T (rounds before the swap) to roughly where Δ levels off.

### 4. Main runs (Saturday)

Freeze the settings first (write them in the report). Then:

```bash
python experiment.py base   --out runs/main --n 15 --rounds 60
# wave 1: only needs model M
python experiment.py branch --out runs/main --rounds 60 --arms A0,A1,A5,A6,A7,A8
# wave 2: needs M and P running
python experiment.py branch --out runs/main --rounds 60 --arms A2,A4
# competitive baseline
python experiment.py base   --out runs/b0 --n 5 --rounds 60 --prefix competitive
python analyze.py --out runs/main --baseline runs/b0
```

Every command can be re-run after a crash: finished games are skipped.

Results: `runs/main/results/summary.md`, CSVs, and figures in `runs/main/figs/`.
Full prompts and responses for every call: `runs/main/logs/*.jsonl`.

### 5. Stop paying

```bash
modal app stop collusion-m      # and collusion-p
```

GPUs also shut down after 10 idle minutes. Check spend in the Modal dashboard after the first run.

## Safety notes

- The agents only output text; the harness turns it into a number. They cannot run code or reach
  anything on your machine.
- Never commit `.env` or `models.json` (both are in `.gitignore`).
- The Modal endpoint is public but requires your API key.

## Follow-up experiments

```bash
python extensions.py seed       --out runs/seeded --reps 3 --rounds 60
python extensions.py transplant --main runs/main --b0 runs/b0 --rounds 60
python extensions.py deviate    --main runs/main --rounds 30
python extensions.py report     --seeded runs/seeded --main runs/main
python extra_analysis.py --out runs/main
```
