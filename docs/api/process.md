# Process Package (`swimrs.process`)

Daily soil water balance engine with typed dataclasses, Numba kernels, and HDF5 `SwimInput` built from `.swim` containers.

## Loop and IO

::: swimrs.process.run_daily_loop
    options:
      show_source: false
      show_signature_annotations: true

::: swimrs.process.step_day
    options:
      show_source: false
      show_signature_annotations: true

::: swimrs.process.DailyOutput
    options:
      show_source: false
      members: true
      show_signature_annotations: true

::: swimrs.process.SwimInput
    options:
      show_source: false
      members: true
      show_signature_annotations: true
      merge_init_into_class: true

::: swimrs.process.build_swim_input
    options:
      show_source: false
      show_signature_annotations: true

## State containers

::: swimrs.process.WaterBalanceState
    options:
      show_source: false
      members: true
      show_signature_annotations: true
      merge_init_into_class: true

::: swimrs.process.FieldProperties
    options:
      show_source: false
      members: true
      show_signature_annotations: true
      merge_init_into_class: true

::: swimrs.process.CalibrationParameters
    options:
      show_source: false
      members: true
      show_signature_annotations: true
      merge_init_into_class: true
