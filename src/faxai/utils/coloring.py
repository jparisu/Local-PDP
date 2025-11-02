"""
Submodule for color utilities.
"""

from __future__ import annotations

from matplotlib.colors import LinearSegmentedColormap, to_rgba


class Color:
    def __init__(self, rgb: tuple[int, int, int]):
        self._rgb = rgb

    @classmethod
    def from_str(cls, color_str: str) -> Color:
        """
        Create a Color instance from a color string, whether named or hexadecimal.

        Args:
            color_str (str): The color string (e.g., "red", "#FF0000").

        Returns:
            Color: An instance of the Color class.
        """
        rgba = to_rgba(color_str)
        rgb = tuple(int(c * 255) for c in rgba[:3])
        return cls(rgb)

    @classmethod
    def from_rgb(cls, r: int, g: int, b: int) -> Color:
        """
        Create a Color instance from RGB values.

        Args:
            r (int): Red component (0-255).
            g (int): Green component (0-255).
            b (int): Blue component (0-255).

        Returns:
            Color: An instance of the Color class.
        """
        return cls((r, g, b))

    def to_hex(self) -> str:
        """
        Convert the Color instance to a hexadecimal color string.

        Returns:
            str: The hexadecimal color string (e.g., "#FF0000").
        """
        return "#{:02X}{:02X}{:02X}".format(*self._rgb)

    def to_rgb(self) -> tuple[int, int, int]:
        """
        Get the RGB values of the Color instance.

        Returns:
            tuple[int, int, int]: The RGB values as a tuple.
        """
        return self._rgb


def transparent_colormap(color: Color, *, n: int = 256) -> LinearSegmentedColormap:
    """
    Create a colormap that transitions from white to the specified base color.

    Args:
        base_color (str): The base color to transition to. Defaults to "red".
        n (int): Number of discrete colors in the colormap. Defaults to 256.
    """
    c_rgb = color.to_rgb()
    return LinearSegmentedColormap.from_list("white-to-color", [(0, (1, 1, 1)), (1, c_rgb)], N=n)
