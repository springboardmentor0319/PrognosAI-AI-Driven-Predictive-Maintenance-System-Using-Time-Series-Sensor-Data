// ── Grafana base URL (host-exposed port, absolute from browser) ────────────
const GRAFANA = 'http://localhost:3100';
const KIOSK   = 'kiosk=tv&theme=dark&orgId=1';

// ── Alpine.js root store ───────────────────────────────────────────────────
function appStore() {
    return {
        activeTab: 'fleet',

        // Engine Deep Dive selections
        dataset:    'FD001',
        engineUnit: 1,

        // ── Grafana iframe URLs ──────────────────────────────────────────
        get fleetUrl() {
            return `${GRAFANA}/d/prognosai-fleet-v2?${KIOSK}`;
        },
        get engineUrl() {
            return `${GRAFANA}/d/prognosai-engine-v2?${KIOSK}&var-dataset=${this.dataset}&var-engine_unit=${this.engineUnit}`;
        },
        get modelUrl() {
            return `${GRAFANA}/d/prognosai-model-v2?${KIOSK}`;
        },

        // ── Tab switching ────────────────────────────────────────────────
        setTab(tab) {
            this.activeTab = tab;
        },

        isActive(tab) {
            return this.activeTab === tab;
        },
    };
}
