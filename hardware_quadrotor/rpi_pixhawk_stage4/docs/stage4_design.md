# F450 Stage 4 real-hardware command framework design

Stage 4 is an independently gated framework for a later, separately approved second flight. This implementation task is code-and-offline-test only.

## Architecture

- `audit-only` is the default. It loads no pymavlink transport module and creates no socket.
- Telemetry remains a receive-only localhost `udpin` endpoint, normally `127.0.0.1:14556` from Rpanion.
- Commands use a separately configured Rpanion UDP Server endpoint, `udpout:127.0.0.1:<operator-port>`. There is no default command port.
- `transport/pymavlink_boundary.py` is the only file permitted to call MAVLink transmit methods. It exposes only PING/TIMESYNC probe, bounded NED velocity, and zero velocity.
- The flight state machine is synchronous and single-threaded. It never arms, takes off, changes mode, lands, writes parameters, or runs motor tests.

## Stop policy

Mode departure from GUIDED or RC override immediately stops all sends and exits without a final zero command, so the process cannot fight the pilot. Stale telemetry or an internal safety fault may attempt at most one bounded zero velocity before closing. Normal completion sends one zero velocity and closes. No automatic recovery is allowed.

## Initial reference and limits

The current hover point is locked as the reference origin. The first two seconds hold zero velocity; the remaining active interval uses a smooth 0.10 m north-axis displacement only. Runtime is capped at five seconds. Limits are 0.10 m/s horizontal, 0.10 m/s vertical, 0.08 rad/s yaw rate, 0.5 m horizontal radius, and 1.5 m relative height.

The existing candidate_097 parameters and Stage 2 control law remain unchanged. Stage 4 applies an additional independent output limiter.
