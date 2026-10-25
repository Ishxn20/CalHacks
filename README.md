# CalHacks

## How we use Convex

The operating-room console runs on Convex: a live instrument board, an event timeline and arm charts.
Recorded hand-off episodes go to Convex file storage, and Claude labels them in a scheduled action.
Crons retry failed labels and prune old readings, and an HTTP action exports the accepted dataset.
See [hackathon.md](hackathon.md) for the features, the files that show them, and the build log.
