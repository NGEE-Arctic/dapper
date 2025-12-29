# Dapper → ELM Smoke-Test Harness (OLMT backend)

**Purpose (north star):**
Validate that *dapper-exported* ELM inputs (met + surface + domain [+ optional landuse]) are **structurally correct** by proving ELM can **initialize, read the files, advance a few timesteps, and exit cleanly**.

This is **dev/test-only** tooling. It is not a user-facing workflow.

---

## 1) What we are testing
**Pass criteria (smoke test):**
- Case is created successfully
- Model executable launches
- ELM reads the provided:
  - `surffile`
  - `domainfile`
  - forcing (`metdir`) in **cpl_bypass** format
  - (optional) `landusefile` if included
- Model advances **≥ 2 timesteps**
- Exit code is **0** (clean termination)

**Not testing:** scientific realism, spinup quality, flux correctness, calibration.

---

## 2) Core decisions (commitments)
1. **Backend:** Use **OLMT** as the runner **now**.
2. **Pin OLMT:** dapper uses a **single pinned OLMT commit** (vendored or submodule). No “fork soup.”
3. **ELM branches:** runner takes an explicit `model_root` pointing at the **ELM/E3SM checkout under test**.
4. **Forcing mode (MVP):** **cpl_bypass** first (because dapper already exports it). DATM can be later.
5. **Short runs:** guarantee “few timesteps” (preferred: CIME STOP_N override after case creation; fallback: minimal OLMT `nyears_*`).
6. **Multi-site testing:** run **one site/gridcell per case** initially (simple attribution). Batch mode later.
7. **Dev-only:** not shipped as “normal dapper usage” and allowed to be opinionated.

---

## 3) Interfaces / contracts
### 3.1 RunSpec (written by dapper)
A small JSON/YAML file adjacent to exported inputs.

**Required fields**
- `inputs.metdir` (cpl_bypass forcing directory)
- `inputs.domainfile` (NetCDF)
- `inputs.surffile` (NetCDF)
- `inputs.landusefile` (optional; may point to a known-good static file for MVP)
- `time.start`, `time.end` (or “enough coverage for N steps”)
- `time.dt_seconds` (or dt hours)
- `site.id` (string identifier; not necessarily an OLMT catalog site)

**Optional fields**
- `elm.namelist_overrides` (key/values or snippets)
- `features` (flags like polygonal tundra / hillslope hydrology etc., for future)

### 3.2 RunnerConfig (dev harness)
- `elm.model_root` (path to E3SM/ELM checkout; branch under test)
- `olmt.root` (path to pinned OLMT)
- `inputdata.root` (ccsm_inputdata)
- `output.caseroot`, `output.runroot`
- `machine` / `compiler` / `mpilib` (as required by OLMT)
- `run.short_run_profile` (nsteps target, etc.)

---

## 4) Execution flow (one RunSpec)
1. **Export inputs** with dapper (met + surf + domain [+ optional landuse])
2. Write **RunSpec**
3. Choose `model_root` (ELM branch under test)
4. Create a dedicated output sandbox: `caseroot/runroot/logroot`
5. Invoke pinned OLMT `site_fullrun.py` with explicit pointers:
   - `--model_root`, `--ccsm_input`, `--caseroot`, `--runroot`
   - `--cpl_bypass --metdir --domainfile --surffile [--landusefile]`
6. **Force short run** (see §5)
7. Run
8. Parse logs and record **PASS/FAIL**
9. Emit an **Artifact Manifest** (see §7)

---

## 5) Short-run guarantee (critical)
**Preferred method (most deterministic):**
- Let OLMT create the case.
- Immediately set CIME run length to tiny values:
  - `STOP_OPTION=nsteps`
  - `STOP_N=4` (or 2–8)
  - `RESUBMIT=0`
- Then run.

**Fallback method (only if STOP_N override isn’t practical initially):**
- Use OLMT phase lengths to minimize runtime:
  - `nyears_ad_spinup = 1`
  - `nyears_final_spinup = 0 or 1`
  - `nyears_transient = 0`
- Ensure forcing “cycle length” doesn’t force longer runs (easiest: export 1-year forcing).

---

## 6) Multi-site / multi-gridcell testing
**MVP:** loop over RunSpecs → **one case per site/gridcell**
- Pros: failures are attributable; logs are clean.
- Cons: slower if you rebuild every time (avoid by caching builds).

**Later:** batch mode (if needed)
- Site-group runs, or “all sites” style loops if supported in your OLMT fork.

---

## 7) Required provenance + artifacts (non-negotiable)
Every run must write a machine-readable `artifact_manifest.json` containing:
- `dapper_version` (or git hash)
- `olmt_commit`
- `elm_commit` (for the model_root checkout)
- hashes of `surffile/domainfile/landusefile` (+ met file listing/hashes)
- the exact OLMT command line used
- paths to `caseroot`, `runroot`, and main logs
- PASS/FAIL + failure reason (first error line, traceback, or nonzero exit)

This is how we avoid “what actually ran?” confusion.

---

## 8) Known risks + mitigations
- **OLMT drift/forks:** mitigate by pinning OLMT commit used by dapper.
- **Landuse dependency:** MVP can use a known-good static landuse; add dapper landuse generation later.
- **Inputdata availability:** require `ccsm_inputdata` path and validate early.
- **Branch-specific features:** treat as RunSpec “profiles” (named configurations) rather than ad-hoc tweaks.

---

## 9) Milestones (pragmatic)
1. **MVP (1 branch, 1 site):** dapper exports + pinned OLMT run + PASS/FAIL
2. **Matrix (N branches × M sites):** cache builds; standardized artifacts
3. **Profiles:** add named RunSpec profiles for branch-specific options
4. **Optional:** DATM backend
5. **Optional:** native CIME backend plugin

---

## 10) When we get lost: reset checklist
Ask these in order:
1. Are we still only proving “ELM can read inputs and take a few steps”?
2. Is OLMT pinned, and are we using the pinned version?
3. Are we testing *dapper-produced* files (not OLMT-regenerated ones)?
4. Is the short-run guarantee in place (STOP_N override or minimal nyears)?
5. Do we have a manifest with commits + file hashes for this run?
6. If a branch fails: is it input-structure, branch feature mismatch, or toolchain/setup?

---

**Status note:** Docker is optional. If it doesn’t include OLMT/E3SM, it can still be used as a toolchain container with bind mounts, but WSL-native execution is the simplest MVP path.
