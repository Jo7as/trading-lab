# Deploying on Railway (optional)

trading-lab runs locally with no infrastructure. If you want an always-on
**evolution worker** (searches parameters 24/7) and/or a **Telegram signal
worker** (posts live signals), the repo ships a ready-made Docker setup for
[Railway](https://railway.app). Railway builds `Dockerfile` automatically;
`railway.toml` pins the builder and a restart policy.

One repository serves **all** worker types. You deploy the same repo twice (or
more) and only change `LAB_WORKER` and the volume.

## 1. Prerequisites

- The repo pushed to GitHub.
- A Railway account.
- For the signal worker: a Telegram bot token and chat id (see §4).

## 2. Create the project

1. Railway → **New Project → Deploy from GitHub repo** → pick your fork.
2. Railway detects `Dockerfile` and builds it. Repeat for a second service
   (same repo) — you will end up with e.g. a `evolution` service and a
   `signal` service.

## 3. The two services

Add a **volume** to each service and set variables. On first boot the
container seeds an empty volume from the matching template.

### Evolution worker

| Setting | Value |
|---|---|
| Volume mount | `/app/state` |
| `LAB_WORKER` | `evolution` (or a strategy worker, see below) |
| `LAB_MODE` | `paper` |
| `LAB_STATE` | `/app/state` |
| `LAB_DATA_SOURCE` | `dukascopy` |
| optional `LAB_STATE_TEMPLATE` | `state-template` \| `jpy-state-template` \| `smc-state-template` \| `orb-state-template` \| `orb2r-state-template` \| `session-range-state-template` \| `mean-reversion-state-template` |

The worker writes `goal.yaml`, `strategy.yaml`, `hypotheses.jsonl`, `best_oos*`
and `history/` into the volume and keeps evolving across restarts.

### Signal worker (Telegram)

| Setting | Value |
|---|---|
| Volume mount | `/app/live-state` |
| `LAB_WORKER` | `signal` |
| `LAB_MODE` | `paper` |
| `LAB_STATE` | `/app/live-state` |
| `LAB_DATA_SOURCE` | `dukascopy` |
| `TELEGRAM_BOT_TOKEN` | from @BotFather |
| `TELEGRAM_CHAT_ID` | your chat/channel id |
| optional `SIGNAL_ASSETS` | `gold` (default) or `gold,jpy` |
| optional `SIGNAL_INTERVAL_SECONDS` | `60` |

The volume is seeded from `live-state-template` (contains `hf_baseline.yaml`).
The gold strategy file is read from `LAB_STRATEGY_PATH`
(default `<LAB_STATE>/hf_baseline.yaml`).

## 4. Telegram setup

1. Talk to **@BotFather** → `/newbot` → copy the **token**.
2. Add the bot to your chat/channel, send it a message, then read the **chat
   id** (e.g. via `https://api.telegram.org/bot<token>/getUpdates`).
3. Set `TELEGRAM_BOT_TOKEN` and `TELEGRAM_CHAT_ID` on the signal service.

Without a token the signal worker still runs, it just has nowhere to post.

## 5. Worker types and their tuning knobs

`LAB_WORKER` selects the worker. Every evolution worker reads
`<PREFIX>_EVOLUTION_*` variables (year range, candidates, rounds, forward
days, asset):

| `LAB_WORKER` | Strategy | Env prefix |
|---|---|---|
| `evolution` | Supertrend + Volume (default) | `EVOLUTION_*` |
| `smc_evolution` | SMC / ICT | `SMC_EVOLUTION_*` |
| `orb_evolution` | Opening-range breakout | `ORB_EVOLUTION_*` |
| `orb2r_evolution` | ORB, fixed R target | `ORB2R_EVOLUTION_*` |
| `session_range_evolution` | Session-range fade | `SRF_EVOLUTION_*` |
| `mr_evolution` | Mean reversion | `MR_EVOLUTION_*` |
| `signal` | replay best strategy | `SIGNAL_*` |

Full variable list: `.env.example` and `trading_lab/run.py`.

## 6. Notes

- State lives only in the volume; the container itself is stateless.
- The historical `HERMES_*` variable names are still accepted, so an env file
  from an older deployment keeps working unchanged (`trading_lab/env.py`).
- `scripts/trading_monitor_check.sh` is an optional health check that talks to
  Railway via the `railway` CLI.
