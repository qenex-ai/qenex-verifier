# Q-Lang Functional Pipe Test
# Goal: Test F#/Elixir style piping for clean data flow

# Define a scalar
define x = 16.0

# Chain: 16 -> sqrt -> sqrt
# Expected: 2.0
x |> sqrt |> sqrt

# Chain with Physics
define speed = 0.5 * c
# Speed -> gamma -> sqrt (just to test chaining math on QValues)
speed |> gamma |> sqrt
