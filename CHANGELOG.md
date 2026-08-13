## [v1.0.0] - 2026-08-12

First stable release. The FastAPI + PostgreSQL REST service under `rest_server/` is now the
supported backend, and its HTTP API is covered by semantic versioning: breaking changes require
a 2.0.0.

### Added
- `rest_server.__version__`, surfaced by `GET /` and the OpenAPI schema, so a deployed instance
  can report its release version.

### Changed
- **Breaking:** the backend moved from Neo4j (v0.2.0) to PostgreSQL, with a new API surface
  under `/v1/assets/*` and the `/modelcards` / `/datasheets` read endpoints. There is no
  automated Neo4j-to-PostgreSQL data migration.

### Removed
- Repo hygiene: dropped a committed demo CSV export (`model_cards 2026-05-17 12-37-00.csv`) and
  working notes (`dev_log.md`) that shouldn't ship in a 1.0.0 tree.

### Known limitations
- `legacy/` (Flask + Neo4j REST server) and `mcp_server/` are retained in-repo for reference
  only and are **not supported** under 1.0.0; bugs filed against them will not be treated as
  1.0.x regressions. CI no longer installs or tests `mcp_server/`'s dependencies.
- `mcp_server/`'s `mcp` dependency has 6 open high-severity Dependabot advisories (WebSocket
  Host/Origin validation, HTTP session-request authentication, DNS rebinding protection). Since
  `mcp_server/` is archive-only and excluded from the 1.0.0 support contract, these are not
  fixed in this release — do not run `mcp_server/` against untrusted networks.
- Model similarity requires an OpenAI API key and is off by default
  (`ENABLE_MC_SIMILARITY=False`).
- Bulk ingest is partial-success: each item in a bulk request is validated and inserted
  independently, so a `200 OK` may still contain per-item errors in `results`.

## [v0.2.0] - 2025-06-10

### Added
- API Endpoints:
  - `/get_github_credentials` (GET): Retrieve GitHub username and token.
  - `/get_huggingface_credentials` (GET): Retrieve Hugging Face username and token.
  - `/modelcard_linkset` (HEAD): Provides model card linkset relations in the HTTP Link header for improved discoverability and interoperability.
- New Project Logo

### Changed
  - Integration guides for OpenAI, Hugging Face, and GitHub.
