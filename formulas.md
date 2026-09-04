# Formulas

The formulas we derive, experiment by experiment. Each is added where we first use it.

## Experiment 1 - Single LIF neuron

The neuron updates its voltage each step, then fires:

```
tau * dV/dt = -V + I          then:  if V >= threshold: fire and reset V to 0
```

**Firing condition** (does it fire at all on a steady current?):

```
fires if  I * tau > threshold      otherwise: silent
```

**Voltage ceiling** (the highest voltage the leak allows):

```
V_inf = I * tau
```

**Spike interval** (time between two spikes):

```
interval = tau * ln( V_inf / (V_inf - threshold) )
```

**Spike count** (over a fixed run of T steps):

```
spikes = T / interval
```

---

## Notation glossary

| Symbol | Meaning | Units in our runs |
|:--|:--|:--|
| `I` | injected input current (steady) | dimensionless |
| `tau` | membrane time constant (how fast the voltage reacts) | time steps |
| `threshold` | voltage that triggers a spike | voltage |
| `V` | membrane voltage (internal state) | voltage |
| `V_inf` | equilibrium voltage ceiling, equal to `I * tau` | voltage |
| `T` | total simulated time | time steps |
| `interval` | time between consecutive spikes | time steps |
| `spikes` | number of spikes in the run | count |
