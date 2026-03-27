// ── Fleet health summary — polls /stats/fleet every 30s ───────────────────
function fleetStore() {
    return {
        total:      '—',
        lastHour:   '—',
        healthy:    '—',
        warning:    '—',
        critical:   '—',
        loading:    true,
        error:      false,

        async fetchFleet() {
            try {
                const res = await fetch('/stats/fleet');
                if (!res.ok) throw new Error(res.statusText);
                const data = await res.json();

                this.total    = data.total_predictions ?? '—';
                this.lastHour = data.predictions_last_hour ?? '—';

                // Reset counts
                this.healthy = 0; this.warning = 0; this.critical = 0;

                for (const row of (data.health_breakdown || [])) {
                    const status = (row.health_status || '').toUpperCase();
                    const count  = Number(row.engine_count || 0);
                    if (status === 'HEALTHY')  this.healthy  += count;
                    if (status === 'WARNING')  this.warning  += count;
                    if (status === 'CRITICAL') this.critical += count;
                }

                this.error   = false;
            } catch (e) {
                this.error = true;
            } finally {
                this.loading = false;
            }
        },

        init() {
            this.fetchFleet();
            setInterval(() => this.fetchFleet(), 30_000);
        },
    };
}
