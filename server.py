"""FastAPI server with SSE streaming for the micro-economy simulation."""

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

from fastapi import FastAPI, Query
from fastapi.responses import HTMLResponse
from scripts.run_dgp import create_dgp_agents
from sse_starlette.sse import EventSourceResponse

from micro_economy.dgp_personas import ALL_ARCHETYPES, sample_population
from micro_economy.experiments import EXPERIMENT_CONFIGS
from micro_economy.metrics import compute_all_metrics
from micro_economy.simulation import create_agents, run_simulation, run_simulation_sync

app = FastAPI(title="Micro-Economy Simulator")

# Track active simulation so we can cancel it when a new one starts
_active_simulation: asyncio.Event | None = None


@app.get("/", response_class=HTMLResponse)
async def index():
    """Serve the frontend."""
    html_path = Path(__file__).parent / "static/index.html"
    return HTMLResponse(html_path.read_text())


@app.get("/api/stream")
async def stream_simulation(
    rounds: int = Query(default=20, ge=1, le=100),
    delay: float = Query(default=1.5, ge=0.1, le=10.0),
):
    """SSE endpoint that streams simulation events."""
    global _active_simulation

    # Cancel any previous simulation
    if _active_simulation is not None:
        _active_simulation.set()

    cancel_event = asyncio.Event()
    _active_simulation = cancel_event

    agents = create_agents()

    async def event_generator():
        async for event in run_simulation(num_rounds=rounds, agents=agents, delay=delay):
            if cancel_event.is_set():
                return
            yield {"event": event["type"], "data": json.dumps(event)}

    return EventSourceResponse(event_generator())


@app.get("/api/config")
async def config():
    """Return current config info."""
    has_key = bool(os.environ.get("OPENAI_API_KEY"))
    return {
        "llm_mode": "openai" if has_key else "mock",
        "model": "gpt-4o-mini" if has_key else "heuristic",
    }


@app.get("/experiments", response_class=HTMLResponse)
async def experiments_page():
    """Serve the experiments UI."""
    html_path = Path(__file__).parent / "static/experiments.html"
    return HTMLResponse(html_path.read_text())


@app.get("/api/experiments/configs")
async def list_configs():
    """List available experiment configurations."""
    return {
        name: [
            {
                "name": p.name,
                "personality": p.personality,
                "risk_tolerance": p.risk_tolerance,
                "production_skill": p.production_skill.value,
            }
            for p in personas
        ]
        for name, personas in EXPERIMENT_CONFIGS.items()
    }


@app.get("/api/experiments/run")
async def run_experiment_stream(
    configs: str = Query(default="all", description="Comma-separated config names or 'all'"),
    runs: int = Query(default=10, ge=1, le=50),
    rounds: int = Query(default=20, ge=1, le=100),
):
    """SSE endpoint that streams experiment progress and results."""
    if configs == "all":
        selected = EXPERIMENT_CONFIGS
    else:
        names = [c.strip() for c in configs.split(",")]
        selected = {k: v for k, v in EXPERIMENT_CONFIGS.items() if k in names}

    async def event_generator():
        all_results = []
        total_runs = len(selected) * runs
        completed = 0

        for config_name, personas in selected.items():
            config_results = []
            for run_id in range(runs):
                # Run in thread pool to avoid blocking
                agents, market = await asyncio.get_event_loop().run_in_executor(
                    None, run_simulation_sync, rounds, personas, 200.0
                )
                m = compute_all_metrics(agents, market)
                m["config"] = config_name
                m["run_id"] = run_id
                config_results.append(m)
                all_results.append(m)
                completed += 1

                yield {
                    "event": "progress",
                    "data": json.dumps(
                        {
                            "config": config_name,
                            "run_id": run_id,
                            "completed": completed,
                            "total": total_runs,
                            "metrics": m,
                        }
                    ),
                }

            # Config complete — send summary
            ginis = [r["gini"] for r in config_results]
            vols = [r["avg_volatility"] for r in config_results]
            trades = [r["trade_volume"]["total_trades"] for r in config_results]

            yield {
                "event": "config_complete",
                "data": json.dumps(
                    {
                        "config": config_name,
                        "summary": {
                            "gini_mean": round(sum(ginis) / len(ginis), 4),
                            "gini_std": round(_std(ginis), 4),
                            "volatility_mean": round(sum(vols) / len(vols), 4),
                            "volatility_std": round(_std(vols), 4),
                            "trades_mean": round(sum(trades) / len(trades), 1),
                            "trades_std": round(_std(trades), 1),
                        },
                        "runs": config_results,
                    }
                ),
            }

        yield {
            "event": "complete",
            "data": json.dumps({"all_results": all_results}),
        }

    return EventSourceResponse(event_generator())


@app.get("/dgp", response_class=HTMLResponse)
async def dgp_page():
    """Serve the DGP simulation UI."""
    html_path = Path(__file__).parent / "static/dgp.html"
    return HTMLResponse(html_path.read_text())


@app.get("/api/dgp/run")
async def run_dgp_stream(
    samples: int = Query(default=10, ge=1, le=50),
    rounds: int = Query(default=20, ge=1, le=100),
    seed: int = Query(default=42),
):
    """SSE endpoint that streams DGP simulation round-by-round."""
    import random as _random

    _random.seed(seed)

    population = sample_population(ALL_ARCHETYPES, samples_per_archetype=samples)
    total_groups = samples

    async def event_generator():
        all_results = []

        # Send init with archetype info
        yield {
            "event": "init",
            "data": json.dumps(
                {
                    "archetypes": [a.archetype for a in ALL_ARCHETYPES],
                    "total_groups": total_groups,
                    "samples": samples,
                    "rounds": rounds,
                }
            ),
        }

        for group_idx in range(total_groups):
            group_features = []
            for arch_idx in range(len(ALL_ARCHETYPES)):
                feature_idx = arch_idx * samples + group_idx
                group_features.append(population[feature_idx])

            agents = create_dgp_agents(group_features)

            # Run simulation and stream each round
            from micro_economy.dgp import dgp_decision
            from micro_economy.models import Good, MarketState
            from micro_economy.simulation import match_orders

            market = MarketState()
            agents_map = {a.name: a for a in agents}

            for agent in agents:
                agent.wealth_history.append(agent.net_worth(market.prices))

            for round_num in range(1, rounds + 1):
                all_orders = []
                for agent in agents:
                    orders = dgp_decision(agent, market, round_num)
                    all_orders.extend(orders)

                for order in all_orders:
                    if order.action == "buy":
                        good = Good(order.good)
                        market.demand[good] = market.demand.get(good, 0) + order.quantity

                buys = [o for o in all_orders if o.action == "buy"]
                sells = [o for o in all_orders if o.action == "sell"]
                trades = match_orders(buys, sells, market, round_num)

                for t in trades:
                    buyer = agents_map[t.buyer]
                    seller = agents_map[t.seller]
                    total_cost = t.price * t.quantity
                    if buyer.budget >= total_cost and seller.inventory.get(t.good, 0) >= t.quantity:
                        buyer.budget -= total_cost
                        seller.budget += total_cost
                        seller.inventory[t.good] = seller.inventory.get(t.good, 0) - t.quantity
                        buyer.inventory[t.good] = buyer.inventory.get(t.good, 0) + t.quantity

                market.trade_log.extend(trades)
                market.update_prices()
                market.record_prices()

                for agent in agents:
                    agent.wealth_history.append(agent.net_worth(market.prices))

                yield {
                    "event": "round",
                    "data": json.dumps(
                        {
                            "group": group_idx,
                            "round": round_num,
                            "prices": market.prices_dict(),
                            "agents": [
                                {
                                    "name": a.name,
                                    "archetype": a.features.archetype,
                                    "wealth": round(a.net_worth(market.prices), 2),
                                    "budget": round(a.budget, 2),
                                    "inventory": {g.value: a.inventory.get(g, 0) for g in Good},
                                }
                                for a in agents
                            ],
                            "trades": len(trades),
                        }
                    ),
                }

                await asyncio.sleep(0.02)  # small delay for streaming

            # Group complete — collect results
            group_results = []
            for agent in agents:
                group_results.append(
                    {
                        "name": agent.name,
                        "archetype": agent.features.archetype,
                        "risk_appetite": round(agent.features.risk_appetite, 4),
                        "patience": round(agent.features.patience, 4),
                        "final_wealth": round(agent.wealth_history[-1], 2),
                        "wealth_trajectory": [round(w, 2) for w in agent.wealth_history],
                    }
                )
            group_results.sort(key=lambda r: r["final_wealth"], reverse=True)
            for rank, r in enumerate(group_results):
                r["rank"] = rank + 1

            all_results.extend(group_results)

            yield {
                "event": "group_complete",
                "data": json.dumps(
                    {
                        "group": group_idx,
                        "results": group_results,
                    }
                ),
            }

        # Final summary by archetype
        from collections import defaultdict

        by_arch = defaultdict(list)
        for r in all_results:
            by_arch[r["archetype"]].append(r)

        summary = {}
        for arch, agents_data in sorted(by_arch.items()):
            wealths = [a["final_wealth"] for a in agents_data]
            ranks = [a["rank"] for a in agents_data]
            w_mean = sum(wealths) / len(wealths)
            w_std = (sum((w - w_mean) ** 2 for w in wealths) / len(wealths)) ** 0.5
            r_mean = sum(ranks) / len(ranks)
            summary[arch] = {
                "wealth_mean": round(w_mean, 2),
                "wealth_std": round(w_std, 2),
                "rank_mean": round(r_mean, 2),
                "count": len(agents_data),
                "trajectories": [a["wealth_trajectory"] for a in agents_data],
            }

        yield {
            "event": "complete",
            "data": json.dumps(
                {
                    "summary": summary,
                    "all_results": all_results,
                }
            ),
        }

    return EventSourceResponse(event_generator())


def _std(vals: list[float]) -> float:
    if len(vals) < 2:
        return 0.0
    m = sum(vals) / len(vals)
    return (sum((v - m) ** 2 for v in vals) / len(vals)) ** 0.5


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("server:app", host="0.0.0.0", port=8765, reload=True)
