-- Create extensions
CREATE EXTENSION IF NOT EXISTS timescaledb CASCADE;
CREATE EXTENSION IF NOT EXISTS vector;

-- Create database user with permissions
DO $$
BEGIN
   IF NOT EXISTS (SELECT FROM pg_catalog.pg_roles WHERE rolname = 'crypto_user') THEN
      CREATE USER crypto_user WITH PASSWORD 'crypto_password';
   END IF;
END $$;

-- Grant privileges
GRANT ALL PRIVILEGES ON DATABASE crypto_trading TO crypto_user;
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO crypto_user;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO crypto_user;

-- Set search path for convenience
ALTER ROLE crypto_user SET search_path TO public;
