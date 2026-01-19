# Legacy and Deprecated

!!! warning "Deprecated"
    The `prep/` and `model/` packages and `SamplePlots` helpers are deprecated and will be removed in a future release. Use `SwimContainer` + `build_swim_input` + `run_daily_loop` instead. See `DEPRECATION_PLAN.md` for migration details.

## Legacy prep/model

::: swimrs.prep
    options:
      show_source: false
      members: false

::: swimrs.model
    options:
      show_source: false
      members: false

## Compatibility helpers

::: swimrs.swim.SamplePlots
    options:
      show_source: false
      members: true
      show_signature_annotations: true

::: swimrs.swim.ContainerPlots
    options:
      show_source: false
      members: true
      show_signature_annotations: true

## Analysis status

`swimrs.analysis` is limited; `compare_etf_estimates` is still used by the CLI for metrics. Other analysis utilities are considered legacy.

::: swimrs.analysis.metrics.compare_etf_estimates
    options:
      show_source: false
      show_signature_annotations: true
