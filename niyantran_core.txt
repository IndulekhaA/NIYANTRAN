"""
NIYANTRAN - Intelligent Demand Coordination for Bengaluru's EV Grid
Full ML System: MARL + Forecasting + Grid Hero Dispatch + Infrastructure Planning
"""

import numpy as np
import pandas as pd
from collections import deque, defaultdict
import random
import math
import json
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────
# 1. SYNTHETIC DATA GENERATOR (Bengaluru Grid)
# ─────────────────────────────────────────────

class BengaluruGridSimulator:
    """
    Simulates Bengaluru's actual distribution network topology.
    Based on BESCOM feeder zones: Whitefield, Koramangala, Indiranagar,
    HSR Layout, Electronic City, Hebbal, Rajajinagar, Jayanagar.
    """
    FEEDER_ZONES = {
        "Whitefield":     {"capacity_kw": 12000, "ev_density": 0.38, "peak_hour": 19},
        "Koramangala":    {"capacity_kw": 9500,  "ev_density": 0.42, "peak_hour": 20},
        "Indiranagar":    {"capacity_kw": 8000,  "ev_density": 0.35, "peak_hour": 20},
        "HSR_Layout":     {"capacity_kw": 10000, "ev_density": 0.45, "peak_hour": 19},
        "Electronic_City":{"capacity_kw": 15000, "ev_density": 0.30, "peak_hour": 18},
        "Hebbal":         {"capacity_kw": 7500,  "ev_density": 0.28, "peak_hour": 21},
        "Rajajinagar":    {"capacity_kw": 9000,  "ev_density": 0.25, "peak_hour": 20},
        "Jayanagar":      {"capacity_kw": 8500,  "ev_density": 0.32, "peak_hour": 20},
    }

    def __init__(self, n_chargers_per_zone=500, seed=42):
        np.random.seed(seed)
        self.n_chargers_per_zone = n_chargers_per_zone
        self.zones = list(self.FEEDER_ZONES.keys())

    def generate_episode(self, hours=24):
        """Generate one day of grid data across all zones."""
        records = []
        base_date = datetime(2024, 6, 1, 0, 0)

        for zone, props in self.FEEDER_ZONES.items():
            cap = props["capacity_kw"]
            ev_d = props["ev_density"]
            peak_h = props["peak_hour"]

            for h in range(hours):
                ts = base_date + timedelta(hours=h)
                # Base load (non-EV) - sinusoidal daily pattern
                base_load = cap * 0.35 + cap * 0.20 * math.sin(math.pi * (h - 6) / 12)

                # EV charging demand - evening peak with spread
                n_active_evs = int(self.n_chargers_per_zone * ev_d *
                                   self._plug_in_probability(h, peak_h))
                ev_load = n_active_evs * np.random.uniform(3.3, 7.4)  # 3.3-7.4 kW per charger

                # Solar generation (if any rooftop)
                solar_gen = cap * 0.08 * max(0, math.sin(math.pi * (h - 6) / 12)) if 6 <= h <= 18 else 0

                net_load = base_load + ev_load - solar_gen
                utilization = net_load / cap
                is_overload = utilization > 0.95

                records.append({
                    "timestamp": ts,
                    "zone": zone,
                    "hour": h,
                    "base_load_kw": round(base_load, 2),
                    "ev_load_kw": round(ev_load, 2),
                    "solar_gen_kw": round(solar_gen, 2),
                    "net_load_kw": round(net_load, 2),
                    "capacity_kw": cap,
                    "utilization": round(utilization, 4),
                    "is_overload": int(is_overload),
                    "n_active_evs": n_active_evs,
                    "temperature_c": 28 + 6 * math.sin(math.pi * (h - 4) / 12) + np.random.normal(0, 1),
                    "day_of_week": ts.weekday(),
                })
        return pd.DataFrame(records)

    def _plug_in_probability(self, hour, peak_h):
        """Gaussian plug-in probability centered at peak hour."""
        return math.exp(-0.5 * ((hour - peak_h) / 2.5) ** 2)

    def generate_dataset(self, n_days=90):
        """Generate 90 days of training data."""
        frames = []
        for day in range(n_days):
            df = self.generate_episode()
            df["timestamp"] = df["timestamp"] + timedelta(days=day)
            df["day"] = day
            frames.append(df)
        return pd.concat(frames, ignore_index=True)


# ─────────────────────────────────────────────
# 2. DEMAND FORECASTING MODEL (LSTM-style)
# ─────────────────────────────────────────────

class DemandForecaster:
    """
    Forecasts per-zone net load 4 hours ahead using temporal features.
    Uses a gradient-boosted ensemble (pure numpy/scipy-free, sklearn-compatible).
    For production: swap with actual LSTM/Transformer.
    """

    def __init__(self, horizon_h=4):
        self.horizon_h = horizon_h
        self.models = {}  # one model per zone
        self.zone_stats = {}

    def _build_features(self, df, zone):
        zdf = df[df["zone"] == zone].sort_values("timestamp").reset_index(drop=True)
        feats = []
        targets = []
        for i in range(12, len(zdf) - self.horizon_h):
            row = zdf.iloc[i]
            hist = zdf.iloc[i-12:i]
            f = [
                row["hour"],
                row["day_of_week"],
                row["temperature_c"],
                row["solar_gen_kw"],
                hist["net_load_kw"].mean(),
                hist["net_load_kw"].std(),
                hist["net_load_kw"].iloc[-1],
                hist["net_load_kw"].iloc[-2],
                hist["ev_load_kw"].mean(),
                hist["n_active_evs"].mean(),
                row["utilization"],
                math.sin(2 * math.pi * row["hour"] / 24),
                math.cos(2 * math.pi * row["hour"] / 24),
            ]
            feats.append(f)
            targets.append(zdf.iloc[i + self.horizon_h]["net_load_kw"])
        return np.array(feats), np.array(targets)

    def train(self, df):
        """Train a simple ridge regression per zone (replace with LGBM/LSTM in prod)."""
        print("Training Demand Forecaster...")
        for zone in df["zone"].unique():
            X, y = self._build_features(df, zone)
            if len(X) < 10:
                continue
            # Normalize
            mu, sigma = X.mean(0), X.std(0) + 1e-8
            Xn = (X - mu) / sigma
            self.zone_stats[zone] = (mu, sigma)
            # Ridge regression: w = (X'X + λI)^{-1} X'y
            lam = 1.0
            A = Xn.T @ Xn + lam * np.eye(Xn.shape[1])
            b = Xn.T @ y
            w = np.linalg.solve(A, b)
            self.models[zone] = w
            pred = Xn @ w
            mae = np.mean(np.abs(pred - y))
            print(f"  {zone}: MAE={mae:.1f} kW")

    def predict(self, zone, features_vec):
        """Predict next horizon_h load for zone."""
        if zone not in self.models:
            return None
        mu, sigma = self.zone_stats[zone]
        xn = (np.array(features_vec) - mu) / sigma
        return float(xn @ self.models[zone])

    def forecast_all_zones(self, current_df):
        """Return forecast dict for all zones from latest data."""
        results = {}
        for zone in current_df["zone"].unique():
            zdf = current_df[current_df["zone"] == zone].sort_values("timestamp")
            if len(zdf) < 13:
                continue
            row = zdf.iloc[-1]
            hist = zdf.iloc[-13:-1]  # 12 rows strictly before last
            if len(hist) < 2:
                continue
            f = [
                row["hour"], row["day_of_week"], row["temperature_c"],
                row["solar_gen_kw"],
                hist["net_load_kw"].mean(), hist["net_load_kw"].std(),
                hist["net_load_kw"].iloc[-1], hist["net_load_kw"].iloc[-2],
                hist["ev_load_kw"].mean(), hist["n_active_evs"].mean(),
                row["utilization"],
                math.sin(2 * math.pi * row["hour"] / 24),
                math.cos(2 * math.pi * row["hour"] / 24),
            ]
            pred = self.predict(zone, f)
            cap = BengaluruGridSimulator.FEEDER_ZONES[zone]["capacity_kw"]
            # Clamp to physically valid range (10% to 130% of capacity)
            pred = float(np.clip(pred, cap * 0.10, cap * 1.30)) if pred is not None else None
            results[zone] = {
                "forecast_kw": round(pred, 1) if pred else None,
                "capacity_kw": cap,
                "forecast_utilization": round(pred / cap, 4) if pred else None,
                "overload_risk": pred > cap * 0.90 if pred else False,
            }
        return results


# ─────────────────────────────────────────────
# 3. MULTI-AGENT REINFORCEMENT LEARNING (MARL)
# ─────────────────────────────────────────────

class ChargerAgent:
    """
    Individual EV charger agent using Q-learning.
    State: (soc_bucket, hour_bucket, zone_utilization_bucket, flexibility_mode)
    Actions: 0=pause, 1=charge_low(1.6kW), 2=charge_mid(3.3kW), 3=charge_full(7.4kW), 4=discharge(V2G)
    """
    ACTIONS = [0, 1.6, 3.3, 7.4, -3.0]  # kW; negative = V2G discharge
    ACTION_NAMES = ["pause", "low", "mid", "full", "v2g_discharge"]

    def __init__(self, agent_id, flexibility_mode="flexible", alpha=0.1, gamma=0.95, epsilon=0.3):
        self.id = agent_id
        self.flexibility_mode = flexibility_mode  # "full_charge", "flexible", "grid_hero"
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.q_table = defaultdict(lambda: np.zeros(len(self.ACTIONS)))
        self.soc = np.random.uniform(0.15, 0.60)  # initial state of charge
        self.target_soc = 0.90
        self.battery_kwh = 40.0  # typical EV
        self.episode_reward = 0
        self.charges_deferred = 0

    def get_state(self, hour, zone_utilization):
        soc_b = min(int(self.soc * 5), 4)  # 0-4
        hour_b = hour // 4  # 0-5
        util_b = min(int(zone_utilization * 4), 3)  # 0-3
        mode_b = {"full_charge": 0, "flexible": 1, "grid_hero": 2}[self.flexibility_mode]
        return (soc_b, hour_b, util_b, mode_b)

    def select_action(self, state):
        # Constraint: can't discharge if not grid_hero
        valid_actions = list(range(len(self.ACTIONS)))
        if self.flexibility_mode != "grid_hero":
            valid_actions = [a for a in valid_actions if self.ACTIONS[a] >= 0]
        # Constraint: can't charge if full
        if self.soc >= 0.99:
            valid_actions = [0]  # pause only

        if np.random.random() < self.epsilon:
            return random.choice(valid_actions)
        q_vals = self.q_table[state].copy()
        q_vals[[a for a in range(len(self.ACTIONS)) if a not in valid_actions]] = -np.inf
        return int(np.argmax(q_vals))

    def compute_reward(self, action_idx, zone_utilization, electricity_price, dt_h=0.25):
        power_kw = self.ACTIONS[action_idx]
        energy_kwh = power_kw * dt_h

        # Update SOC
        delta_soc = energy_kwh / self.battery_kwh
        new_soc = np.clip(self.soc + delta_soc, 0.0, 1.0)

        # Rewards
        r_soc = 3.0 * (new_soc - self.soc) if power_kw >= 0 else 0  # reward charging
        r_soc_penalty = -5.0 if new_soc < 0.20 else 0  # penalize low SOC
        r_grid = -4.0 * zone_utilization * abs(power_kw) / 7.4 if power_kw > 0 else 0
        r_price = -electricity_price * energy_kwh * 0.1 if power_kw > 0 else 0
        r_v2g = 2.0 * abs(power_kw) / 7.4 if power_kw < 0 and zone_utilization > 0.85 else 0
        r_full = 5.0 if new_soc >= self.target_soc and self.soc < self.target_soc else 0

        # Full-charge mode: never pause unless grid crisis
        if self.flexibility_mode == "full_charge" and power_kw == 0 and zone_utilization < 0.95:
            r_grid -= 3.0

        reward = r_soc + r_soc_penalty + r_grid + r_price + r_v2g + r_full
        self.soc = new_soc
        return reward

    def learn(self, state, action, reward, next_state):
        current_q = self.q_table[state][action]
        next_q = np.max(self.q_table[next_state])
        self.q_table[state][action] += self.alpha * (
            reward + self.gamma * next_q - current_q
        )
        self.episode_reward += reward

    def decay_epsilon(self, min_eps=0.05, decay=0.995):
        self.epsilon = max(min_eps, self.epsilon * decay)


class NiyantranMARL:
    """
    Multi-Agent RL coordinator.
    Coordinates agents per zone. No central command — coordination
    emerges from shared grid signal (zone utilization).
    """
    def __init__(self, n_agents_per_zone=100):
        self.n_agents = n_agents_per_zone
        self.zones = list(BengaluruGridSimulator.FEEDER_ZONES.keys())
        self.agents = {}
        self.zone_utilization = {z: 0.5 for z in self.zones}
        self.training_log = []

        # Initialize agents per zone with mode distribution
        for zone in self.zones:
            zone_agents = []
            for i in range(n_agents_per_zone):
                # Mode distribution: 30% full_charge, 50% flexible, 20% grid_hero
                r = random.random()
                if r < 0.30:
                    mode = "full_charge"
                elif r < 0.80:
                    mode = "flexible"
                else:
                    mode = "grid_hero"
                zone_agents.append(ChargerAgent(f"{zone}_{i}", flexibility_mode=mode))
            self.agents[zone] = zone_agents

    def step_zone(self, zone, hour, zone_utilization, electricity_price=6.0):
        """Run one timestep for all agents in a zone."""
        agents = self.agents[zone]
        total_power = 0.0
        actions_taken = defaultdict(int)

        for agent in agents:
            state = agent.get_state(hour, zone_utilization)
            action = agent.select_action(state)
            reward = agent.compute_reward(action, zone_utilization, electricity_price)
            next_state = agent.get_state(hour, agent.soc)
            agent.learn(state, action, reward, next_state)
            total_power += ChargerAgent.ACTIONS[action]
            actions_taken[ChargerAgent.ACTION_NAMES[action]] += 1

        return total_power, dict(actions_taken)

    def train(self, n_episodes=200, hours_per_episode=24):
        """Train all agents over simulated episodes."""
        print(f"\nTraining MARL system: {n_episodes} episodes × {hours_per_episode}h")
        sim = BengaluruGridSimulator(n_chargers_per_zone=self.n_agents)

        for ep in range(n_episodes):
            ep_data = sim.generate_episode(hours_per_episode)
            ep_overloads = 0
            ep_total_power = 0

            for hour in range(hours_per_episode):
                for zone, props in BengaluruGridSimulator.FEEDER_ZONES.items():
                    hour_data = ep_data[(ep_data["zone"] == zone) & (ep_data["hour"] == hour)]
                    if hour_data.empty:
                        continue
                    row = hour_data.iloc[0]
                    # Grid signal: current utilization (shared, not commands)
                    base_util = row["utilization"]
                    price = 8.0 if 18 <= hour <= 22 else 4.5  # ToU pricing

                    total_power, actions = self.step_zone(zone, hour, base_util, price)
                    ep_total_power += abs(total_power)

                    # Simulated post-coordination utilization
                    cap = props["capacity_kw"]
                    adjusted_load = row["base_load_kw"] + max(0, total_power)
                    new_util = adjusted_load / cap
                    if new_util > 0.95:
                        ep_overloads += 1

            # Decay exploration
            for zone in self.zones:
                for agent in self.agents[zone]:
                    agent.decay_epsilon()

            if ep % 20 == 0:
                avg_soc = np.mean([a.soc for z in self.zones for a in self.agents[z]])
                print(f"  Episode {ep:3d}: overloads={ep_overloads}, avg_soc={avg_soc:.2f}")
                self.training_log.append({
                    "episode": ep, "overloads": ep_overloads,
                    "avg_soc": round(avg_soc, 3), "total_power_kw": round(ep_total_power, 1)
                })

        print("MARL training complete.")

    def get_zone_summary(self):
        """Return per-zone agent statistics."""
        summary = {}
        for zone in self.zones:
            agents = self.agents[zone]
            modes = defaultdict(int)
            socs = []
            for a in agents:
                modes[a.flexibility_mode] += 1
                socs.append(a.soc)
            grid_heroes = [a for a in agents if a.flexibility_mode == "grid_hero"]
            v2g_capacity = len(grid_heroes) * 3.0  # kW discharge per grid hero
            summary[zone] = {
                "n_agents": len(agents),
                "mode_distribution": dict(modes),
                "avg_soc": round(np.mean(socs), 3),
                "grid_heroes": len(grid_heroes),
                "v2g_capacity_kw": round(v2g_capacity, 1),
            }
        return summary


# ─────────────────────────────────────────────
# 4. GRID HERO DISPATCH ENGINE
# ─────────────────────────────────────────────

class GridHeroDispatch:
    """
    When feeder stress is detected, dispatch Grid Hero vehicles
    to inject power back (V2G) and shift non-hero loads.
    Treats 70,000 Grid Hero participants as a virtual battery.
    """
    def __init__(self, total_grid_heroes=70000, avg_battery_kwh=40.0, v2g_power_kw=3.0):
        self.total_heroes = total_grid_heroes
        self.avg_battery_kwh = avg_battery_kwh
        self.v2g_power_kw = v2g_power_kw
        self.dispatch_log = []

    def total_v2g_capacity_kw(self, available_fraction=0.60):
        """Available V2G power from Grid Hero fleet."""
        return self.total_heroes * available_fraction * self.v2g_power_kw

    def dispatch(self, zone, forecast_utilization, capacity_kw, timestamp):
        """
        Determine dispatch action for a zone.
        Returns dispatch decision with expected load reduction.
        """
        if forecast_utilization < 0.80:
            return {"zone": zone, "action": "none", "load_reduction_kw": 0}

        overload_kw = (forecast_utilization - 0.90) * capacity_kw
        if overload_kw <= 0:
            overload_kw = 0

        # Strategy 1: Pause non-critical charging
        pause_power = capacity_kw * 0.05  # 5% load shift

        # Strategy 2: V2G from Grid Heroes in zone
        # Assume zone has proportional grid heroes
        zone_heroes = int(self.total_heroes / 8)  # 8 zones
        available_soc_heroes = int(zone_heroes * 0.60)  # 60% available
        v2g_power = available_soc_heroes * self.v2g_power_kw

        # Strategy 3: Defer flexible chargers
        defer_power = capacity_kw * 0.08

        total_reduction = pause_power + v2g_power + defer_power
        severity = "critical" if forecast_utilization > 0.95 else "warning"

        decision = {
            "zone": zone,
            "timestamp": str(timestamp),
            "forecast_utilization": round(forecast_utilization, 3),
            "severity": severity,
            "action": "dispatch",
            "pause_kw": round(pause_power, 1),
            "v2g_kw": round(v2g_power, 1),
            "defer_kw": round(defer_power, 1),
            "total_reduction_kw": round(total_reduction, 1),
            "expected_utilization_post": round(
                (forecast_utilization * capacity_kw - total_reduction) / capacity_kw, 3
            ),
            "grid_heroes_activated": available_soc_heroes,
        }
        self.dispatch_log.append(decision)
        return decision

    def dispatch_all_zones(self, forecasts, timestamp=None):
        """Run dispatch for all zones based on forecasts."""
        if timestamp is None:
            timestamp = datetime.now()
        decisions = {}
        for zone, fc in forecasts.items():
            if fc["forecast_utilization"] is not None:
                decisions[zone] = self.dispatch(
                    zone, fc["forecast_utilization"],
                    fc["capacity_kw"], timestamp
                )
        return decisions

    def virtual_battery_stats(self):
        """Stats on the Grid Hero virtual battery."""
        total_kwh = self.total_heroes * self.avg_battery_kwh * 0.60  # usable
        max_discharge_kw = self.total_v2g_capacity_kw()
        return {
            "total_grid_heroes": self.total_heroes,
            "virtual_battery_kwh": round(total_kwh, 0),
            "max_v2g_power_kw": round(max_discharge_kw, 0),
            "equivalent_substations": round(max_discharge_kw / 40000, 2),  # 40MW substation
            "co2_offset_tons_per_year": round(max_discharge_kw * 0.3 * 365 / 1000, 0),
        }


# ─────────────────────────────────────────────
# 5. INFRASTRUCTURE PLANNING MODEL
# ─────────────────────────────────────────────

class InfrastructurePlanner:
    """
    Ranks feeder zones by investment priority.
    Combines: demand growth trajectory, spare capacity, EV adoption rate,
    feeder stress frequency, and investment efficiency.
    Produces a continuously updated investment priority map.
    """

    def __init__(self, projection_years=5):
        self.projection_years = projection_years
        self.ev_growth_rate = 0.35  # 35% YoY EV adoption growth in Bengaluru
        self.infra_cost_per_mw = 8_500_000  # INR per MW upgrade

    def compute_demand_growth(self, df, zone):
        """Estimate demand growth rate from historical data."""
        zdf = df[df["zone"] == zone].groupby("day")["net_load_kw"].max().reset_index()
        if len(zdf) < 2:
            return 0.02
        x = np.arange(len(zdf))
        y = zdf["net_load_kw"].values
        # Linear regression slope
        slope = np.polyfit(x, y, 1)[0]
        return slope / (y.mean() + 1e-8)  # normalized daily growth rate

    def project_utilization(self, current_util, growth_rate, years):
        """Project utilization over years (capped at 1.5 to avoid overflow)."""
        annual_growth = min(growth_rate * 365, 0.40)  # cap at 40% annual growth
        return min(current_util * (1 + annual_growth) ** years, 1.5)

    def score_zone(self, zone, df, current_forecasts):
        props = BengaluruGridSimulator.FEEDER_ZONES[zone]
        cap = props["capacity_kw"]
        ev_density = props["ev_density"]

        # Current state
        zdf = df[df["zone"] == zone]
        current_util = zdf["utilization"].tail(24).mean()
        overload_freq = zdf["is_overload"].mean()
        growth_rate = self.compute_demand_growth(df, zone)

        # Future projection
        util_5yr = self.project_utilization(current_util, growth_rate, self.projection_years)

        # Spare capacity (how much headroom before overload)
        spare_capacity_kw = max(0, cap * 0.95 - current_util * cap)

        # Investment efficiency: problems prevented per INR spent
        # Overloads prevented = projected annual overloads if no action
        projected_annual_overloads = overload_freq * 365 * 24
        upgrade_cost = max(spare_capacity_kw, 1000) * self.infra_cost_per_mw / 1000
        efficiency = projected_annual_overloads / (upgrade_cost / 1_000_000 + 1e-8)

        # Composite priority score (higher = invest sooner)
        score = (
            0.30 * min(util_5yr, 1.5) +          # future load pressure
            0.25 * overload_freq * 10 +            # current stress frequency
            0.20 * ev_density +                    # EV adoption (demand driver)
            0.15 * growth_rate * 100 +             # demand growth velocity
            0.10 * (1 - spare_capacity_kw / cap)   # spare capacity tightness
        )

        fc = current_forecasts.get(zone, {})
        return {
            "zone": zone,
            "priority_score": round(score, 4),
            "current_utilization": round(current_util, 3),
            "projected_5yr_utilization": round(min(util_5yr, 2.0), 3),
            "overload_frequency": round(overload_freq, 4),
            "demand_growth_rate_daily": round(growth_rate, 6),
            "spare_capacity_kw": round(spare_capacity_kw, 1),
            "ev_density": ev_density,
            "forecast_overload_risk": fc.get("overload_risk", False),
            "recommended_upgrade_kw": round(max(0, util_5yr * cap - cap * 0.90), 0),
            "estimated_cost_inr": round(max(0, util_5yr * cap - cap * 0.90) * self.infra_cost_per_mw / 1000, 0),
            "investment_efficiency": round(efficiency, 4),
        }

    def rank_zones(self, df, current_forecasts):
        """Rank all zones by investment priority."""
        scores = []
        for zone in BengaluruGridSimulator.FEEDER_ZONES:
            scores.append(self.score_zone(zone, df, current_forecasts))
        ranked = sorted(scores, key=lambda x: x["priority_score"], reverse=True)
        for i, r in enumerate(ranked):
            r["rank"] = i + 1
        return ranked


# ─────────────────────────────────────────────
# 6. REAL-TIME FEEDER MONITOR
# ─────────────────────────────────────────────

class FeederMonitor:
    """
    Watches all feeder zones in real time.
    Triggers dispatch before spikes form.
    Maintains rolling window of utilization history.
    """
    ALERT_THRESHOLDS = {"warning": 0.80, "critical": 0.90, "emergency": 0.95}

    def __init__(self, window_hours=4):
        self.window_hours = window_hours
        self.history = defaultdict(lambda: deque(maxlen=window_hours * 4))  # 15-min intervals
        self.alerts = []

    def update(self, zone, utilization, timestamp):
        self.history[zone].append({"ts": timestamp, "util": utilization})

    def get_trend(self, zone):
        """Detect if utilization is accelerating upward."""
        h = list(self.history[zone])
        if len(h) < 4:
            return "stable"
        utils = [x["util"] for x in h]
        recent = np.mean(utils[-4:])
        earlier = np.mean(utils[:-4]) if len(utils) > 4 else recent
        delta = recent - earlier
        if delta > 0.10:
            return "rising_fast"
        elif delta > 0.05:
            return "rising"
        elif delta < -0.05:
            return "falling"
        return "stable"

    def check_alerts(self, zone, utilization, timestamp):
        trend = self.get_trend(zone)
        for level, thresh in sorted(self.ALERT_THRESHOLDS.items(),
                                    key=lambda x: x[1], reverse=True):
            if utilization >= thresh:
                alert = {
                    "zone": zone, "timestamp": str(timestamp),
                    "level": level, "utilization": round(utilization, 3),
                    "trend": trend, "pre_emptive": trend == "rising_fast"
                }
                self.alerts.append(alert)
                return alert
        return None


# ─────────────────────────────────────────────
# 7. NIYANTRAN ORCHESTRATOR (Full System)
# ─────────────────────────────────────────────

class Niyantran:
    """
    Full system orchestrator integrating all components.
    """
    def __init__(self, n_agents_per_zone=100, train_marl_episodes=100):
        print("=" * 60)
        print("  NIYANTRAN — Intelligent Demand Coordination")
        print("  Bengaluru EV Grid Intelligence Platform")
        print("=" * 60)

        self.sim = BengaluruGridSimulator(n_chargers_per_zone=n_agents_per_zone)
        self.forecaster = DemandForecaster(horizon_h=4)
        self.marl = NiyantranMARL(n_agents_per_zone=n_agents_per_zone)
        self.dispatch = GridHeroDispatch(total_grid_heroes=70000)
        self.planner = InfrastructurePlanner()
        self.monitor = FeederMonitor()

        print("\n[1/3] Generating synthetic Bengaluru grid data (90 days)...")
        self.dataset = self.sim.generate_dataset(n_days=90)
        print(f"      {len(self.dataset):,} records across {len(self.sim.zones)} zones")

        print("\n[2/3] Training Demand Forecaster...")
        self.forecaster.train(self.dataset)

        print(f"\n[3/3] Training MARL Coordination System ({train_marl_episodes} episodes)...")
        self.marl.train(n_episodes=train_marl_episodes)

        print("\n✓ Niyantran fully initialized.\n")

    def run_real_time_step(self, current_data=None):
        """Simulate one real-time coordination step."""
        if current_data is None:
            current_data = self.dataset  # use full dataset for context

        ts = datetime.now()
        hour = ts.hour

        # 1. Forecast
        forecasts = self.forecaster.forecast_all_zones(current_data)

        # 2. Monitor & alert
        alerts = {}
        for zone, fc in forecasts.items():
            util = fc["forecast_utilization"] or 0
            self.monitor.update(zone, util, ts)
            alert = self.monitor.check_alerts(zone, util, ts)
            if alert:
                alerts[zone] = alert

        # 3. Dispatch Grid Heroes if needed
        dispatch_decisions = self.dispatch.dispatch_all_zones(forecasts, ts)

        # 4. MARL agent actions (sample one timestep)
        marl_actions = {}
        for zone, props in BengaluruGridSimulator.FEEDER_ZONES.items():
            fc = forecasts.get(zone, {})
            util = fc.get("forecast_utilization") or 0.5
            price = 8.0 if 18 <= hour <= 22 else 4.5
            total_power, actions = self.marl.step_zone(zone, hour, util, price)
            marl_actions[zone] = {
                "total_coordinated_power_kw": round(total_power, 1),
                "action_distribution": actions,
            }

        # 5. Infrastructure planning
        infra_rankings = self.planner.rank_zones(current_data, forecasts)

        return {
            "timestamp": str(ts),
            "forecasts": forecasts,
            "alerts": alerts,
            "dispatch_decisions": dispatch_decisions,
            "marl_coordination": marl_actions,
            "infrastructure_rankings": infra_rankings,
        }

    def print_dashboard(self, result):
        """Pretty-print the real-time dashboard."""
        print("\n" + "=" * 60)
        print(f"  NIYANTRAN LIVE DASHBOARD  |  {result['timestamp'][:19]}")
        print("=" * 60)

        print("\n📡 ZONE FORECASTS (4h ahead):")
        print(f"  {'Zone':<20} {'Forecast kW':>12} {'Utilization':>12} {'Risk':>8}")
        print("  " + "-" * 56)
        for zone, fc in result["forecasts"].items():
            risk = "🔴 HIGH" if fc["overload_risk"] else "🟢 OK"
            print(f"  {zone:<20} {str(fc['forecast_kw']):>12} {str(fc['forecast_utilization']):>12} {risk:>8}")

        if result["alerts"]:
            print(f"\n⚠️  ALERTS ({len(result['alerts'])} zones):")
            for zone, alert in result["alerts"].items():
                print(f"  [{alert['level'].upper()}] {zone}: util={alert['utilization']}, trend={alert['trend']}")

        print(f"\n⚡ GRID HERO DISPATCH:")
        dispatched = {z: d for z, d in result["dispatch_decisions"].items()
                      if d["action"] == "dispatch"}
        if dispatched:
            for zone, d in dispatched.items():
                print(f"  {zone}: V2G={d['v2g_kw']}kW, pause={d['pause_kw']}kW, "
                      f"reduce={d['total_reduction_kw']}kW → util {d['forecast_utilization']}→{d['expected_utilization_post']}")
        else:
            print("  No dispatch needed — grid healthy")

        print(f"\n🏗️  INFRASTRUCTURE PRIORITY RANKING:")
        for r in result["infrastructure_rankings"][:5]:
            print(f"  #{r['rank']} {r['zone']:<20} score={r['priority_score']:.3f} "
                  f"5yr_util={r['projected_5yr_utilization']:.2f} "
                  f"upgrade={r['recommended_upgrade_kw']:.0f}kW")

        vb = self.dispatch.virtual_battery_stats()
        print(f"\n🔋 VIRTUAL BATTERY (Grid Hero Fleet):")
        print(f"  {vb['total_grid_heroes']:,} heroes → {vb['virtual_battery_kwh']:,.0f} kWh usable")
        print(f"  Max V2G: {vb['max_v2g_power_kw']:,.0f} kW  |  CO₂ offset: {vb['co2_offset_tons_per_year']:,.0f} T/yr")

        print("\n" + "=" * 60)

    def full_report(self):
        """Run and return complete system report."""
        result = self.run_real_time_step()
        self.print_dashboard(result)
        return result

    def save_results(self, result, path="niyantran_output.json"):
        """Save results to JSON."""
        # Convert numpy types
        def convert(obj):
            if isinstance(obj, (np.integer, np.int64)):
                return int(obj)
            if isinstance(obj, (np.floating, np.float64)):
                return float(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, bool):
                return bool(obj)
            return obj

        def deep_convert(d):
            if isinstance(d, dict):
                return {k: deep_convert(v) for k, v in d.items()}
            if isinstance(d, list):
                return [deep_convert(i) for i in d]
            return convert(d)

        with open(path, "w") as f:
            json.dump(deep_convert(result), f, indent=2)
        print(f"\n✓ Results saved to {path}")


# ─────────────────────────────────────────────
# 8. ENTRY POINT
# ─────────────────────────────────────────────

if __name__ == "__main__":
    # Full system — reduce episodes for quick demo (set to 500+ for production)
    system = Niyantran(n_agents_per_zone=50, train_marl_episodes=50)

    print("\nRunning real-time coordination step...")
    result = system.full_report()
    system.save_results(result)

    print("\n📊 MARL Training Log:")
    for entry in system.marl.training_log[-5:]:
        print(f"  Episode {entry['episode']}: overloads={entry['overloads']}, avg_soc={entry['avg_soc']}")

    print("\n📋 Agent Mode Distribution (Whitefield zone sample):")
    zone_summary = system.marl.get_zone_summary()
    for zone, s in list(zone_summary.items())[:3]:
        print(f"  {zone}: heroes={s['grid_heroes']}, v2g_cap={s['v2g_capacity_kw']}kW, avg_soc={s['avg_soc']}")
