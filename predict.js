// ── Sensor sets per subset (mirrors config.py SENSORS) ────────────────────
const SENSORS = {
    1: ['s2','s3','s4','s7','s8','s9','s11','s12','s13','s14','s17','s20','s21'],
    2: ['s2','s3','s4','s7','s8','s9','s11','s12','s13','s14','s15','s17','s20','s21'],
    3: ['s2','s3','s4','s7','s8','s9','s11','s12','s13','s14','s15','s17','s20','s21'],
    4: ['s2','s3','s4','s7','s8','s9','s11','s12','s13','s14','s15','s17','s20','s21'],
};

// Human-readable sensor labels
const SENSOR_LABELS = {
    s2:  's2 — Fan Inlet Temp (°R)',
    s3:  's3 — HPC Outlet Temp (°R)',
    s4:  's4 — LPT Outlet Temp (°R)',
    s7:  's7 — HPC Outlet Pressure (psia)',
    s8:  's8 — Fan Inlet Flow (lbm/s)',
    s9:  's9 — Core Speed (rpm)',
    s11: 's11 — HPC Outlet Static P (psia)',
    s12: 's12 — Ratio Fan P / T (—)',
    s13: 's13 — Corrected Core Speed (rpm)',
    s14: 's14 — Bypass Ratio (—)',
    s15: 's15 — Bleed Enthalpy (—)',
    s17: 's17 — HP Turbine Cool Air Flow (lbm/s)',
    s20: 's20 — LP Turbine Cool Air Flow (lbm/s)',
    s21: 's21 — Fan Speed Demand (rpm)',
};

function predictStore() {
    return {
        // Form fields
        subset:    1,
        engineId:  1,
        cycle:     100,
        setting1:  0,
        setting2:  0,
        setting3:  100,
        sensors:   {},   // { s2: 0, s3: 0, ... }

        // Result
        result:    null,
        loading:   false,
        error:     null,

        // ── Derived ─────────────────────────────────────────────────────
        get sensorList() {
            return SENSORS[this.subset] || SENSORS[1];
        },
        get showS15() {
            return this.subset !== 1;
        },
        get sensorLabels() {
            return SENSOR_LABELS;
        },
        get healthClass() {
            if (!this.result) return '';
            const s = this.result.health_status;
            if (s === 'HEALTHY')  return 'badge-healthy';
            if (s === 'WARNING')  return 'badge-warning';
            if (s === 'CRITICAL') return 'badge-critical';
            return 'badge-unknown';
        },
        get rulColor() {
            if (!this.result) return '#9ca3af';
            const rul = this.result.predicted_rul;
            if (rul <= 10) return '#ef4444';
            if (rul <= 30) return '#f59e0b';
            return '#22c55e';
        },

        // ── Initialise sensor fields ─────────────────────────────────────
        init() {
            this.resetSensors();
        },

        resetSensors() {
            const fields = {};
            Object.keys(SENSOR_LABELS).forEach(s => { fields[s] = ''; });
            this.sensors = fields;
        },

        onSubsetChange() {
            this.result = null;
            this.error  = null;
        },

        // ── Submit prediction ────────────────────────────────────────────
        async submit() {
            this.loading = true;
            this.error   = null;
            this.result  = null;

            const payload = {
                subset:    Number(this.subset),
                engine_id: Number(this.engineId),
                cycle:     Number(this.cycle),
                setting_1: Number(this.setting1),
                setting_2: Number(this.setting2),
                setting_3: Number(this.setting3),
            };

            // Add only the sensors required for this subset
            for (const s of SENSORS[this.subset]) {
                const val = this.sensors[s];
                if (val === '' || val === null || val === undefined) {
                    this.error   = `Please fill in sensor ${s}.`;
                    this.loading = false;
                    return;
                }
                payload[s] = Number(val);
            }

            try {
                const res = await fetch('/predict', {
                    method:  'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body:    JSON.stringify(payload),
                });
                if (!res.ok) {
                    const err = await res.json().catch(() => ({}));
                    throw new Error(err.detail || `HTTP ${res.status}`);
                }
                this.result = await res.json();
            } catch (e) {
                this.error = e.message;
            } finally {
                this.loading = false;
            }
        },
    };
}
