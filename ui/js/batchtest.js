// ── Batch prediction from test dataset ──────────────────────────────────────
function batchTestStore() {
    return {
        // shared
        dataset:     'FD001',
        // mode: 'snapshot' | 'history' | 'fleet'
        mode:        'fleet',

        // snapshot mode
        maxEngines:     10,
        engines:        [],      // [{unit_number, last_cycle, selected}]
        loadingEngines: false,
        running:        false,
        results:        [],

        // history mode
        demoEngines:     [],  // [{unit_number, max_cycle, total_cycles, selected}]
        loadingDemo:     false,
        runningHistory:  false,
        historyResult:   null,
        historyError:    null,

        // fleet simulation mode
        simulatingFleet: false,
        fleetResult:     null,
        fleetError:      null,

        error: null,

        // ── computed ────────────────────────────────────────────────────────
        get selectedIds() {
            return this.engines.filter(e => e.selected).map(e => e.unit_number);
        },
        get selectedCount() {
            return this.engines.filter(e => e.selected).length;
        },
        get allSelected() {
            return this.engines.length > 0 && this.engines.every(e => e.selected);
        },
        get hasEngines() {
            return this.engines.length > 0;
        },

        get demoSelectedIds() {
            return this.demoEngines.filter(e => e.selected).map(e => e.unit_number);
        },
        get demoSelectedCount() {
            return this.demoEngines.filter(e => e.selected).length;
        },
        get demoAllSelected() {
            return this.demoEngines.length > 0 && this.demoEngines.every(e => e.selected);
        },

        get healthClass() {
            return status => {
                if (status === 'HEALTHY')  return 'text-green-400';
                if (status === 'WARNING')  return 'text-yellow-400';
                if (status === 'CRITICAL') return 'text-red-400';
                return 'text-gray-400';
            };
        },

        get healthBadge() {
            return status => {
                if (status === 'HEALTHY')  return 'bg-green-900/40 text-green-300 border border-green-700/40';
                if (status === 'WARNING')  return 'bg-yellow-900/40 text-yellow-300 border border-yellow-700/40';
                if (status === 'CRITICAL') return 'bg-red-900/40 text-red-300 border border-red-700/40';
                return 'bg-gray-800 text-gray-400';
            };
        },

        // ── initialise ──────────────────────────────────────────────────────
        init() {
            this.loadDemoEngines();
        },

        // ── mode switch ─────────────────────────────────────────────────────
        switchMode(m) {
            this.mode          = m;
            this.results       = [];
            this.historyResult = null;
            this.fleetResult   = null;
            this.error         = null;
            this.historyError  = null;
            this.fleetError    = null;
            if ((m === 'history' || m === 'fleet') && this.demoEngines.length === 0) {
                this.loadDemoEngines();
            } else if (m === 'snapshot' && this.engines.length === 0) {
                this.loadEngines();
            }
        },

        onDatasetChange() {
            this.engines     = [];
            this.demoEngines = [];
            this.results     = [];
            this.historyResult = null;
            this.fleetResult   = null;
            if (this.mode === 'snapshot') {
                this.loadEngines();
            } else {
                this.loadDemoEngines();
            }
        },

        // ── snapshot mode ───────────────────────────────────────────────────
        async loadEngines() {
            this.loadingEngines = true;
            this.engines = [];
            this.results = [];
            this.error   = null;
            try {
                const res = await fetch(`/test-data/${this.dataset}/engines`);
                if (!res.ok) throw new Error(`HTTP ${res.status}`);
                const data = await res.json();
                this.engines = data.engines.map(e => ({ ...e, selected: false }));
            } catch (e) {
                this.error = e.message;
            } finally {
                this.loadingEngines = false;
            }
        },

        toggleAll() {
            const next = !this.allSelected;
            this.engines.forEach(e => e.selected = next);
        },

        selectRandom() {
            this.engines.forEach(e => e.selected = false);
            const shuffled = [...this.engines].sort(() => Math.random() - 0.5);
            shuffled.slice(0, this.maxEngines).forEach(e => {
                const orig = this.engines.find(x => x.unit_number === e.unit_number);
                if (orig) orig.selected = true;
            });
        },

        async runPredictions() {
            this.running = true;
            this.error   = null;
            this.results = [];

            const ids = this.selectedIds;
            const body = ids.length > 0
                ? { dataset: this.dataset, engine_ids: ids,  max_engines: 100 }
                : { dataset: this.dataset, engine_ids: null, max_engines: this.maxEngines };

            try {
                const res = await fetch('/predict/from-test', {
                    method:  'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body:    JSON.stringify(body),
                });
                if (!res.ok) {
                    const err = await res.json().catch(() => ({}));
                    throw new Error(err.detail || `HTTP ${res.status}`);
                }
                const data = await res.json();
                this.results = data.predictions || [];
            } catch (e) {
                this.error = e.message;
            } finally {
                this.running = false;
            }
        },

        // ── demo engine helpers (shared by history + fleet) ─────────────────
        async loadDemoEngines() {
            this.loadingDemo  = true;
            this.demoEngines  = [];
            this.historyError = null;
            this.fleetError   = null;
            try {
                const res = await fetch(`/demo-data/${this.dataset}/engines`);
                if (!res.ok) throw new Error(`HTTP ${res.status}`);
                const data = await res.json();
                this.demoEngines = data.engines.map(e => ({ ...e, selected: false }));
            } catch (e) {
                this.historyError = e.message;
                this.fleetError   = e.message;
            } finally {
                this.loadingDemo = false;
            }
        },

        toggleDemoAll() {
            const next = !this.demoAllSelected;
            this.demoEngines.forEach(e => e.selected = next);
        },

        // ── history mode ────────────────────────────────────────────────────
        async runHistory() {
            this.runningHistory = true;
            this.historyError   = null;
            this.historyResult  = null;

            const ids  = this.demoSelectedIds;
            const body = { dataset: this.dataset, engine_ids: ids.length > 0 ? ids : null };

            try {
                const res = await fetch('/predict/engine-history', {
                    method:  'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body:    JSON.stringify(body),
                });
                if (!res.ok) {
                    const err = await res.json().catch(() => ({}));
                    throw new Error(err.detail || `HTTP ${res.status}`);
                }
                this.historyResult = await res.json();
            } catch (e) {
                this.historyError = e.message;
            } finally {
                this.runningHistory = false;
            }
        },

        // ── fleet simulation mode ────────────────────────────────────────────
        async simulateFleet() {
            this.simulatingFleet = true;
            this.fleetError      = null;
            this.fleetResult     = null;

            const ids  = this.demoSelectedIds;
            const body = { dataset: this.dataset, engine_ids: ids.length > 0 ? ids : null };

            try {
                const res = await fetch('/predict/fleet-snapshot', {
                    method:  'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body:    JSON.stringify(body),
                });
                if (!res.ok) {
                    const err = await res.json().catch(() => ({}));
                    throw new Error(err.detail || `HTTP ${res.status}`);
                }
                this.fleetResult = await res.json();
            } catch (e) {
                this.fleetError = e.message;
            } finally {
                this.simulatingFleet = false;
            }
        },
    };
}
