# Camera interface

The interface controls the viewing direction of the whole scene. It does not bind a drawn line to a selected object.

## Choose the motion source

Select **Pan or tilt path** to draw a camera view path. A horizontal line requests pan. A vertical line requests tilt. A line with movement in both directions requests combined pan and tilt. The full sequence of points controls the request over time. The direction label is a summary of that path, rather than a replacement for it.

Select **Zoom in** or **Zoom out** for focal zoom. The canvas becomes gray and cannot accept drawing input. The backend sends the zoom request without a drawn path. Zoom and a drawn pan or tilt path are not combined in this interface.

A camera view moving left generally makes the visible scene move right. A view moving upward generally makes the visible scene move down. These are camera commands, so they differ from asking an object to move across a fixed background.

## Generate and review

Upload a photograph, select the saved scene, choose the motion type and strength, then press **Generate**. A path needs at least two points. **Cancel** stops the active generation. **Refresh** updates progress. The last generated video has its own playback and downloads.

The strength changes the requested geometry. It does not guarantee that the model follows an exact angle or zoom ratio. The saved samples and measurements show the achieved motion in each tested case.

## Optional scene description

Advanced settings contain a scene description, seed, denoising steps and checkpoint information. The description supplies text context about visible content. It is optional. When it is empty, the interface uses the neutral caption from the evaluated recipe. Camera motion comes from the selected numeric control, rather than from this text.

## Evidence and limits

Horizontal and vertical browser drawings were read from the synchronized canvas and converted to camera requests. Zoom produces no drawn request and disables pointer input. The retained model has been tested on a small set of photographs and motions. Complex curves, independent object paths and physical camera translation remain outside the established result.
