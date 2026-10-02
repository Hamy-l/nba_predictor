"""NBA Predictor application core.

Layering (outermost first):

    web tier         app.py, blueprints/          HTTP framing only
    services         nba_core/services/           workflow orchestration
    serving          nba_core/serving/            scoring + batch execution
    evidence         nba_core/serving/evidence    envelope resolution + cache
    inference        nba_core/inference/          gateway transport
    configuration    constants.py                 keys, URLs, catalogues

Import direction is strictly one-way: outer layers depend on inner ones, never
the reverse.
"""

__version__ = "2.0.0"
