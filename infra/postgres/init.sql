-- ChurnBench: Postgres warehouse schema
-- Historical/analytical layer for the simulated SAM data fabric.
-- Star-schema-ish: fact tables reference dimension tables.
-- The generator (churnbench/generator/) is the authoritative source of data.
-- Nothing is inserted here at boot; init just establishes shape.

CREATE SCHEMA IF NOT EXISTS sam;

-- ─── Dimensions ─────────────────────────────────────────────────────────────

CREATE TABLE sam.dim_vendor (
    vendor_id      SERIAL PRIMARY KEY,
    vendor_name    TEXT NOT NULL UNIQUE,
    vendor_tier    TEXT NOT NULL       -- 'strategic' | 'preferred' | 'tail'
);

CREATE TABLE sam.dim_cost_center (
    cost_center_id SERIAL PRIMARY KEY,
    cost_center    TEXT NOT NULL UNIQUE,
    business_unit  TEXT NOT NULL
);

CREATE TABLE sam.dim_product (
    product_id     SERIAL PRIMARY KEY,
    product_sku    TEXT NOT NULL UNIQUE,
    product_name   TEXT NOT NULL,
    vendor_id      INT REFERENCES sam.dim_vendor(vendor_id),
    license_model  TEXT NOT NULL       -- 'user' | 'device' | 'concurrent' | 'consumption'
);

CREATE TABLE sam.dim_date (
    date_id        DATE PRIMARY KEY,
    fiscal_quarter TEXT NOT NULL,
    fiscal_year    INT NOT NULL
);

-- ─── Facts ──────────────────────────────────────────────────────────────────

CREATE TABLE sam.fact_license_purchase (
    purchase_id    BIGSERIAL PRIMARY KEY,
    product_id     INT REFERENCES sam.dim_product(product_id),
    cost_center_id INT REFERENCES sam.dim_cost_center(cost_center_id),
    purchase_date  DATE NOT NULL,
    seats          INT NOT NULL,
    unit_price_usd NUMERIC(12,2) NOT NULL,
    contract_id    TEXT,               -- ties to Mongo contracts collection
    valid_from     DATE NOT NULL,
    valid_until    DATE NOT NULL
);

CREATE TABLE sam.fact_consumption_event (
    event_id       BIGSERIAL PRIMARY KEY,
    product_id     INT REFERENCES sam.dim_product(product_id),
    user_ext_id    TEXT NOT NULL,      -- ties to Mongo users
    event_date     DATE NOT NULL,
    session_minutes INT NOT NULL,
    api_calls      INT NOT NULL
);

-- Indexes for common analytical queries
CREATE INDEX idx_purchase_date       ON sam.fact_license_purchase(purchase_date);
CREATE INDEX idx_purchase_cc         ON sam.fact_license_purchase(cost_center_id);
CREATE INDEX idx_consumption_date    ON sam.fact_consumption_event(event_date);
CREATE INDEX idx_consumption_product ON sam.fact_consumption_event(product_id);
