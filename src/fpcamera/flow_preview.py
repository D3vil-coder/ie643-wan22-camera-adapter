"""CPU-only preview of the actual first-frame crop and requested camera geometry."""

from __future__ import annotations

from io import BytesIO
from xml.sax.saxutils import escape
import base64

import numpy as np
from PIL import Image, ImageOps

from .flow_geometry import VideoContract, project_points


def conditioning_crop(image: Image.Image, contract: VideoContract) -> Image.Image:
    """Return the same aspect-fill crop used by I2V before first-frame encoding."""
    if not isinstance(image, Image.Image):
        raise TypeError("image must be a PIL Image")
    return ImageOps.fit(
        image.convert("RGB"),
        (contract.width, contract.height),
        method=Image.Resampling.LANCZOS,
    )


def geometry_preview_svg(
    image: Image.Image,
    contract: VideoContract,
    homographies,
    *,
    mode: str,
    thumbnail_width: int = 720,
) -> tuple[Image.Image, str]:
    """Return conditioning crop and self-contained SVG arrows for scene-point flow.

    Arrows show requested first-to-last scene-point displacement from the camera
    homographies. They are a geometry preview, not a generated-video measurement.
    """
    matrices = np.asarray(homographies, dtype=np.float64)
    if matrices.shape != (contract.frames, 3, 3) or not np.isfinite(matrices).all():
        raise ValueError("Expected finite homographies [contract.frames,3,3]")
    if not np.allclose(matrices[0], np.eye(3), atol=1e-6):
        raise ValueError("First-frame homography must be identity")
    if np.any(np.abs(np.linalg.det(matrices)) < 1e-10):
        raise ValueError("Homographies must be nonsingular")
    if thumbnail_width < 160:
        raise ValueError("thumbnail_width must be at least 160")

    crop = conditioning_crop(image, contract)
    thumb_height = max(1, round(thumbnail_width * contract.height / contract.width))
    thumbnail = crop.resize((thumbnail_width, thumb_height), Image.Resampling.LANCZOS)
    buffer = BytesIO()
    thumbnail.save(buffer, format="PNG", optimize=True)
    image_uri = "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode(
        "ascii"
    )

    x_positions = np.asarray([0.22, 0.5, 0.78]) * (contract.width - 1)
    y_positions = np.asarray([0.24, 0.5, 0.76]) * (contract.height - 1)
    points = np.asarray(
        [(x, y) for y in y_positions for x in x_positions], dtype=np.float64
    )
    projected = project_points(matrices[[0, -1]], points)
    scale_x = thumbnail_width / contract.width
    scale_y = thumb_height / contract.height

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{thumbnail_width}" height="{thumb_height}" '
        f'viewBox="0 0 {thumbnail_width} {thumb_height}" role="img" '
        f'aria-label="Geometry preview only; arrows show requested scene-point displacement for {escape(mode)}">',
        '<defs><marker id="flow-arrow" viewBox="0 0 10 10" refX="8" refY="5" '
        'markerWidth="5" markerHeight="5" orient="auto-start-reverse">'
        '<path d="M 0 0 L 10 5 L 0 10 z" fill="#ffb000"/></marker></defs>',
        f'<image x="0" y="0" width="{thumbnail_width}" height="{thumb_height}" '
        f'href="{image_uri}" preserveAspectRatio="none"/>',
    ]
    for start, end in zip(projected[0], projected[1]):
        x0, y0 = float(start[0] * scale_x), float(start[1] * scale_y)
        x1, y1 = float(end[0] * scale_x), float(end[1] * scale_y)
        if np.hypot(x1 - x0, y1 - y0) > 0.05:
            parts.append(
                f'<line x1="{x0:.2f}" y1="{y0:.2f}" x2="{x1:.2f}" y2="{y1:.2f}" '
                'stroke="#ffb000" stroke-width="2.4" marker-end="url(#flow-arrow)"/>'
            )
        parts.append(
            f'<circle cx="{x0:.2f}" cy="{y0:.2f}" r="3.1" '
            'fill="#27d8ff" stroke="#002b3a" stroke-width="1"/>'
        )
    parts.append("</svg>")
    return crop, "".join(parts)
