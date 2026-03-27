-- schema.sql — PostgreSQL schema for the Engine RUL Prediction System
-- Run once against your database:
--   psql -U postgres -d prognosai -f schema.sql
--
-- Tables:
--   engines            — one row per physical engine
--   sensor_readings    — one row per (engine, cycle) measurement
--   model_registry     — trained model metadata; active flag drives prediction logging
--   predictions        — every API prediction, linked to engine + model
--   prediction_errors  — ground truth submitted after the fact; drives accuracy metrics
-- Views:
--   v_fleet_health_summary — live health breakdown for Grafana

-- =============================================================================
-- Extensions
-- =============================================================================
CREATE EXTENSION IF NOT EXISTS pgcrypto;   -- for gen_random_uuid() if needed

-- =============================================================================
-- engines
-- =============================================================================
CREATE TABLE IF NOT EXISTS engines (
    engine_id   SERIAL       PRIMARY KEY,
    unit_number INTEGER      NOT NULL,
    subset      SMALLINT     NOT NULL CHECK (subset BETWEEN 1 AND 4),
    created_at  TIMESTAMPTZ  NOT NULL DEFAULT NOW(),

    CONSTRAINT uq_engines_unit_subset UNIQUE (unit_number, subset)
);

CREATE INDEX IF NOT EXISTS idx_engines_subset ON engines (subset);

-- =============================================================================
-- sensor_readings
-- =============================================================================
CREATE TABLE IF NOT EXISTS sensor_readings (
    reading_id   BIGSERIAL    PRIMARY KEY,
    engine_id    INTEGER      NOT NULL REFERENCES engines (engine_id) ON DELETE CASCADE,
    cycle        INTEGER      NOT NULL CHECK (cycle >= 1),
    -- operating settings
    setting_1    REAL,
    setting_2    REAL,
    setting_3    REAL,
    -- informative sensors (zero-variance sensors excluded)
    s2           REAL,
    s3           REAL,
    s4           REAL,
    s7           REAL,
    s8           REAL,
    s9           REAL,
    s11          REAL,
    s12          REAL,
    s13          REAL,
    s14          REAL,
    s15          REAL,        -- NULL for FD001 (not informative for that subset)
    s17          REAL,
    s20          REAL,
    s21          REAL,
    recorded_at  TIMESTAMPTZ  NOT NULL DEFAULT NOW(),

    CONSTRAINT uq_sensor_readings_engine_cycle UNIQUE (engine_id, cycle)
);

CREATE INDEX IF NOT EXISTS idx_sensor_readings_engine ON sensor_readings (engine_id);
CREATE INDEX IF NOT EXISTS idx_sensor_readings_cycle  ON sensor_readings (engine_id, cycle DESC);

-- =============================================================================
-- model_registry
-- =============================================================================
CREATE TABLE IF NOT EXISTS model_registry (
    model_id         SERIAL       PRIMARY KEY,
    subset           SMALLINT     NOT NULL CHECK (subset BETWEEN 1 AND 4),
    description      TEXT,
    is_active        BOOLEAN      NOT NULL DEFAULT FALSE,
    trained_at       TIMESTAMPTZ  NOT NULL DEFAULT NOW(),
    metrics          JSONB,       -- {"RMSE": 12.3, "MAE": 9.1, "R2": 0.87, "NASA_Score": 345.0}
    -- columns for Dashboard 3 Panel 8 (populated by register_models.py at startup)
    model_version    TEXT,
    algorithm        TEXT         DEFAULT 'XGBoost',
    train_rmse       REAL,
    train_mae        REAL,
    train_r2         REAL,
    train_nasa_score REAL
);

CREATE INDEX IF NOT EXISTS idx_model_registry_subset_active
    ON model_registry (subset, is_active, trained_at DESC);

-- Seed one active model per subset so the API can log predictions.
-- Re-run after retraining (set is_active = FALSE on old rows, INSERT new ones).
INSERT INTO model_registry (subset, description, is_active, algorithm) VALUES
    (1, 'XGBoost FD001 — initial', TRUE, 'XGBoost'),
    (2, 'XGBoost FD002 — initial', TRUE, 'XGBoost'),
    (3, 'XGBoost FD003 — initial', TRUE, 'XGBoost'),
    (4, 'XGBoost FD004 — initial', TRUE, 'XGBoost')
ON CONFLICT DO NOTHING;

-- =============================================================================
-- predictions
-- =============================================================================
CREATE TABLE IF NOT EXISTS predictions (
    prediction_id    BIGSERIAL    PRIMARY KEY,
    request_id       UUID         NOT NULL,          -- groups batch requests
    engine_id        INTEGER      NOT NULL REFERENCES engines (engine_id) ON DELETE CASCADE,
    model_id         INTEGER      NOT NULL REFERENCES model_registry (model_id),
    cycle            INTEGER      NOT NULL CHECK (cycle >= 1),
    n_history_cycles INTEGER      NOT NULL DEFAULT 1,
    predicted_rul    REAL         NOT NULL,
    health_status    TEXT         GENERATED ALWAYS AS (
                         CASE WHEN predicted_rul <= 10 THEN 'CRITICAL'
                              WHEN predicted_rul <= 30 THEN 'WARNING'
                              ELSE 'HEALTHY'
                         END
                     ) STORED,
    rul_cap          SMALLINT     NOT NULL,
    endpoint         TEXT         NOT NULL,          -- /predict | /predict/batch | /predict/sequence
    latency_ms       REAL,
    predicted_at     TIMESTAMPTZ  NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_predictions_engine       ON predictions (engine_id, predicted_at DESC);
CREATE INDEX IF NOT EXISTS idx_predictions_predicted_at ON predictions (predicted_at DESC);
CREATE INDEX IF NOT EXISTS idx_predictions_request      ON predictions (request_id);
CREATE INDEX IF NOT EXISTS idx_predictions_health       ON predictions (health_status);

-- =============================================================================
-- prediction_errors
-- =============================================================================
CREATE TABLE IF NOT EXISTS prediction_errors (
    error_id       BIGSERIAL    PRIMARY KEY,
    prediction_id  BIGINT       NOT NULL REFERENCES predictions (prediction_id) ON DELETE CASCADE,
    true_rul       REAL         NOT NULL CHECK (true_rul >= 0),
    error          REAL         NOT NULL,   -- predicted_rul - true_rul  (positive = late)
    abs_error      REAL         NOT NULL,
    nasa_penalty   REAL         NOT NULL,
    submitted_at   TIMESTAMPTZ  NOT NULL DEFAULT NOW(),

    CONSTRAINT uq_prediction_errors_pred UNIQUE (prediction_id)
);

CREATE INDEX IF NOT EXISTS idx_prediction_errors_pred ON prediction_errors (prediction_id);

-- =============================================================================
-- get_or_create_engine() — used by app.py for engine upsert
-- (also defined in create_function.sql; included here for completeness)
-- =============================================================================
CREATE OR REPLACE FUNCTION get_or_create_engine(
    p_unit_number INTEGER,
    p_subset      SMALLINT
) RETURNS INTEGER
LANGUAGE plpgsql AS $$
DECLARE
    v_engine_id INTEGER;
BEGIN
    SELECT engine_id INTO v_engine_id
    FROM   engines
    WHERE  unit_number = p_unit_number AND subset = p_subset;

    IF NOT FOUND THEN
        INSERT INTO engines (unit_number, subset)
        VALUES (p_unit_number, p_subset)
        RETURNING engine_id INTO v_engine_id;
    END IF;

    RETURN v_engine_id;
END;
$$;

-- =============================================================================
-- v_fleet_health_summary — used by /stats/fleet endpoint and Grafana
-- =============================================================================
CREATE OR REPLACE VIEW v_fleet_health_summary AS
SELECT
    e.subset,
    CASE
        WHEN p.predicted_rul <= 10  THEN 'CRITICAL'
        WHEN p.predicted_rul <= 30  THEN 'WARNING'
        ELSE                             'HEALTHY'
    END                                   AS health_status,
    COUNT(DISTINCT e.engine_id)           AS engine_count,
    ROUND(AVG(p.predicted_rul)::NUMERIC, 1) AS avg_predicted_rul,
    ROUND(MIN(p.predicted_rul)::NUMERIC, 1) AS min_predicted_rul
FROM engines e
JOIN LATERAL (
    -- latest prediction per engine
    SELECT predicted_rul
    FROM   predictions
    WHERE  engine_id = e.engine_id
    ORDER  BY predicted_at DESC
    LIMIT  1
) p ON TRUE
GROUP BY e.subset, health_status
ORDER BY e.subset, health_status;

-- =============================================================================
-- Migration guards — safe to run on existing volumes (ADD COLUMN IF NOT EXISTS)
-- =============================================================================
-- predictions.health_status (generated column — safe to skip if already exists)
DO $$ BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_name='predictions' AND column_name='health_status'
    ) THEN
        ALTER TABLE predictions ADD COLUMN health_status TEXT
            GENERATED ALWAYS AS (
                CASE WHEN predicted_rul <= 10 THEN 'CRITICAL'
                     WHEN predicted_rul <= 30 THEN 'WARNING'
                     ELSE 'HEALTHY'
                END
            ) STORED;
        CREATE INDEX IF NOT EXISTS idx_predictions_health ON predictions (health_status);
    END IF;
END $$;

ALTER TABLE model_registry ADD COLUMN IF NOT EXISTS model_version    TEXT;
ALTER TABLE model_registry ADD COLUMN IF NOT EXISTS algorithm        TEXT DEFAULT 'XGBoost';
ALTER TABLE model_registry ADD COLUMN IF NOT EXISTS train_rmse       REAL;
ALTER TABLE model_registry ADD COLUMN IF NOT EXISTS train_mae        REAL;
ALTER TABLE model_registry ADD COLUMN IF NOT EXISTS train_r2         REAL;
ALTER TABLE model_registry ADD COLUMN IF NOT EXISTS train_nasa_score REAL;
