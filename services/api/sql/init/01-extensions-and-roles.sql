-- Runs once, on first container start, as the superuser.
--
-- Two things happen here that cannot be done from application migrations:
-- installing extensions, and creating the least-privilege roles. The roles
-- matter because they are a security control, not a convenience — see
-- docs/08-DATA-MODEL.md §7.

-- --------------------------------------------------------------------------
-- Extensions
-- --------------------------------------------------------------------------
CREATE EXTENSION IF NOT EXISTS vector;      -- pgvector: embeddings + HNSW
CREATE EXTENSION IF NOT EXISTS citext;      -- case-insensitive email
CREATE EXTENSION IF NOT EXISTS pgcrypto;    -- gen_random_uuid()
CREATE EXTENSION IF NOT EXISTS pg_trgm;     -- fuzzy entity resolution
CREATE EXTENSION IF NOT EXISTS btree_gin;

-- --------------------------------------------------------------------------
-- Roles
-- --------------------------------------------------------------------------

-- The application role. Subject to row-level security; holds no DDL rights, so
-- a compromised application cannot drop a policy to escape its own tenancy.
DO $$
BEGIN
  IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'resx_app') THEN
    CREATE ROLE resx_app LOGIN PASSWORD 'change-me' NOSUPERUSER NOCREATEDB NOCREATEROLE;
  END IF;
END
$$;

-- Handed to the resx-postgres MCP server, so a model-generated query is
-- PHYSICALLY incapable of writing. This is a stronger guarantee than parsing
-- the statement and hoping the parser has no gaps — and we do both.
DO $$
BEGIN
  IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'resx_readonly') THEN
    CREATE ROLE resx_readonly LOGIN PASSWORD 'change-me' NOSUPERUSER NOCREATEDB NOCREATEROLE;
  END IF;
END
$$;

GRANT CONNECT ON DATABASE resx TO resx_app, resx_readonly;
GRANT USAGE ON SCHEMA public TO resx_app, resx_readonly;

-- A generated query must not be able to sit on a connection forever.
ALTER ROLE resx_readonly SET statement_timeout = '10s';
ALTER ROLE resx_app      SET statement_timeout = '30s';

-- Default privileges for tables the migration role creates later. Without
-- these, every migration would have to remember to grant, and one that forgets
-- produces a runtime permission error in production.
ALTER DEFAULT PRIVILEGES FOR ROLE resx_migrate IN SCHEMA public
  GRANT SELECT, INSERT, UPDATE, DELETE ON TABLES TO resx_app;
ALTER DEFAULT PRIVILEGES FOR ROLE resx_migrate IN SCHEMA public
  GRANT SELECT ON TABLES TO resx_readonly;
ALTER DEFAULT PRIVILEGES FOR ROLE resx_migrate IN SCHEMA public
  GRANT USAGE, SELECT ON SEQUENCES TO resx_app;
