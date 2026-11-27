from __future__ import annotations

from functools import singledispatch
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import xarray as xr
from anndata import AnnData
from dask.dataframe import DataFrame as DaskDataFrame
from geopandas import GeoDataFrame
from shapely import MultiPolygon, Point, Polygon
from xarray import DataArray, DataTree

from spatialdata._core.operations.transform import transform
from spatialdata.models import get_axes_names, get_table_keys
from spatialdata.models._utils import SpatialElement
from spatialdata.models.models import Labels2DModel, Labels3DModel, PointsModel, TableModel, get_model
from spatialdata.transformations.operations import get_transformation
from spatialdata.transformations.transformations import BaseTransformation

if TYPE_CHECKING:
    from spatialdata._core.spatialdata import SpatialData

BoundingBoxDescription = dict[str, tuple[float, float]]


def _validate_coordinate_system(e: SpatialElement, coordinate_system: str) -> None:
    d = get_transformation(e, get_all=True)
    assert isinstance(d, dict)
    if coordinate_system not in d:
        raise ValueError(
            f"No transformation to coordinate system {coordinate_system!r} is available for the given element. "
            f"Available coordinate systems: {list(d.keys())}"
        )


def _area_scale_factor(e: SpatialElement, coordinate_system: str) -> float:
    """Factor by which areas (2D) or volumes (3D) of ``e`` scale when mapped into ``coordinate_system``.

    For an affine map, every area/volume scales by ``|det|`` of its linear part, regardless of position.
    """
    t = get_transformation(e, coordinate_system)
    assert isinstance(t, BaseTransformation)
    axes = get_axes_names(e)
    matrix = t.to_affine_matrix(input_axes=axes, output_axes=axes)
    return float(abs(np.linalg.det(matrix[:-1, :-1])))


@singledispatch
def get_centroids(
    e: SpatialElement,
    coordinate_system: str = "global",
    return_background: bool = False,
    return_area: bool = False,
) -> DaskDataFrame:
    """
    Get the centroids of the geometries contained in a SpatialElement, as a new Points element.

    Parameters
    ----------
    e
        The SpatialElement. Only points, shapes (circles, polygons and multipolygons) and labels are supported.
    coordinate_system
        The coordinate system in which the centroids are computed.
    return_background
        If True, the centroid of the background label (0) is included in the output (labels only).
    return_area
        If True, an ``area`` column with the size of each geometry, measured in ``coordinate_system``, is added.
        Labels are measured by counting pixels, circles as ``pi * radius**2`` and polygons by their geometric
        area; the result is then scaled by the absolute determinant of the linear part of the transformation to
        ``coordinate_system``. For 3D labels the voxels are counted, so ``area`` is a volume (as in
        ``skimage.measure.regionprops``). Not supported for points.

    Returns
    -------
    The centroids, as a points element (Dask DataFrame) in the given coordinate system.

    Notes
    -----
    For :class:`~shapely.MultiPolygon` geometries, the centroid is the geometric (area-weighted) centroid computed by
    ``geopandas`` (``GeoDataFrame.centroid``), i.e. sub-polygons contribute in proportion to their area. For
    multiscale labels, centroids and sizes are computed on the full-resolution ``scale0`` level.
    """
    raise ValueError(f"The object type {type(e)} is not supported.")


def _get_centroids_for_labels(xdata: xr.DataArray) -> tuple[pd.DataFrame, np.ndarray]:
    """
    Compute centroids and pixel (voxel) counts for all labels in a DataArray in a single O(n_voxels) pass.

    Works for any number of spatial dimensions (2D and 3D labels).
    """
    arr = xdata.data.compute()
    axes = list(xdata.dims)

    # Map label values to a contiguous range for bincount efficiency.
    label_ids, inverse = np.unique(arr, return_inverse=True)
    flat_inverse = inverse.ravel()
    counts = np.bincount(flat_inverse)  # per-label pixel counts

    # indexing="ij" (matrix convention) ensures the i-th grid varies along the i-th
    # dimension of the output, correctly aligning with xdata.dims for any number of axes.
    coord_grids = np.meshgrid(*[xdata[ax].to_numpy() for ax in axes], indexing="ij")
    data: dict[str, np.ndarray] = {}
    for ax, grid in zip((str(ax) for ax in axes), coord_grids, strict=True):
        coord_sums = np.bincount(flat_inverse, weights=grid.ravel().astype(float, copy=False))
        data[ax] = coord_sums / counts  # counts > 0 by construction (unique guarantees this)

    return pd.DataFrame(data, index=label_ids), counts


def _points_from_centroids(df: pd.DataFrame, e: SpatialElement, coordinate_system: str) -> DaskDataFrame:
    """Build a Points element from the intrinsic centroids of ``e``, transformed into ``coordinate_system``.

    An ``area`` column, if present, holds intrinsic sizes and is scaled into ``coordinate_system`` as well.
    """
    if "area" in df.columns:
        df = df.assign(area=df["area"] * _area_scale_factor(e, coordinate_system))
    t = get_transformation(e, coordinate_system)
    assert isinstance(t, BaseTransformation)
    points = PointsModel.parse(df, transformations={coordinate_system: t})
    transformed: DaskDataFrame = transform(points, to_coordinate_system=coordinate_system)
    return transformed


def _get_intrinsic_centroids_for_labels(
    e: DataArray | DataTree, return_background: bool, return_area: bool
) -> tuple[pd.DataFrame, DataArray]:
    """Intrinsic centroids (and ``area``) of a Labels element, plus the ``scale0`` raster they were measured on."""
    model = get_model(e)
    if model not in [Labels2DModel, Labels3DModel]:
        raise ValueError(
            f"Centroids are not supported for elements modeled by {model.__name__}; "
            f"expected a Labels, Shapes or Points element."
        )
    if isinstance(e, DataTree):
        scale0 = e["scale0"]
        assert isinstance(scale0, DataTree)
        assert len(scale0) == 1
        variable = next(iter(scale0.values()))
        assert isinstance(variable, DataArray)
        e = variable

    df, counts = _get_centroids_for_labels(e)
    if return_area:
        df["area"] = counts.astype(float)
    if not return_background and 0 in df.index:
        df = df.drop(index=0)  # drop the background label
    return df, e


@get_centroids.register(DataArray)
@get_centroids.register(DataTree)
def _(
    e: DataArray | DataTree,
    coordinate_system: str = "global",
    return_background: bool = False,
    return_area: bool = False,
) -> DaskDataFrame:
    """Get the centroids of a Labels element (2D or 3D)."""
    _validate_coordinate_system(e, coordinate_system)
    df, scale0 = _get_intrinsic_centroids_for_labels(e, return_background, return_area)
    return _points_from_centroids(df, scale0, coordinate_system)


def _get_intrinsic_centroids_for_shapes(e: GeoDataFrame, return_area: bool) -> pd.DataFrame:
    """Intrinsic centroids (and ``area``) of a Shapes element (circles or polygons/multipolygons)."""
    # separate points from (multi-)polygons
    first_geometry = e["geometry"].iloc[0]
    if isinstance(first_geometry, Point):
        xy = e.geometry.get_coordinates().values
    else:
        assert isinstance(first_geometry, Polygon | MultiPolygon), (
            f"Expected a GeoDataFrame either composed entirely of circles (Points with the `radius` column) or"
            f" Polygons/MultiPolygons. Found {type(first_geometry)} instead."
        )
        xy = e.centroid.get_coordinates().to_numpy()
    xy_df = pd.DataFrame(xy, columns=["x", "y"], index=e.index.copy())
    if return_area:
        # circles are Point geometries (shapely area 0); their size is carried by the radius column
        is_circle = isinstance(first_geometry, Point)
        area = np.pi * e["radius"].to_numpy(dtype=float) ** 2 if is_circle else e.geometry.area.to_numpy()
        xy_df["area"] = area
    return xy_df


@get_centroids.register(GeoDataFrame)
def _(
    e: GeoDataFrame,
    coordinate_system: str = "global",
    return_background: bool = False,
    return_area: bool = False,
) -> DaskDataFrame:
    """Get the centroids of a Shapes element (circles or polygons/multipolygons)."""
    _validate_coordinate_system(e, coordinate_system)
    return _points_from_centroids(_get_intrinsic_centroids_for_shapes(e, return_area), e, coordinate_system)


def _get_intrinsic_centroids_for_points(e: DaskDataFrame, return_area: bool) -> pd.DataFrame:
    """Intrinsic coordinates of a Points element (points are their own centroids)."""
    if return_area:
        raise ValueError("`return_area` is not supported for points elements (points have no area).")
    axes = get_axes_names(e)
    if axes not in [("x", "y"), ("x", "y", "z")]:
        raise ValueError(f"Expected points axes to be ('x', 'y') or ('x', 'y', 'z'), got {axes}.")
    coords: pd.DataFrame = e[list(axes)].compute()
    return coords


@get_centroids.register(DaskDataFrame)
def _(
    e: DaskDataFrame,
    coordinate_system: str = "global",
    return_background: bool = False,
    return_area: bool = False,
) -> DaskDataFrame:
    """Get the centroids of a Points element."""
    _validate_coordinate_system(e, coordinate_system)
    return _points_from_centroids(_get_intrinsic_centroids_for_points(e, return_area), e, coordinate_system)


def _get_intrinsic_centroids(e: SpatialElement, return_area: bool) -> pd.DataFrame:
    """Intrinsic (untransformed) centroids of ``e``, indexed by instance id, without the background label.

    The columns are ``x, y[, z]`` (in this order, whatever the element's axis order), plus ``area`` if requested.
    """
    if isinstance(e, DataArray | DataTree):
        df, _ = _get_intrinsic_centroids_for_labels(e, return_background=False, return_area=return_area)
    elif isinstance(e, GeoDataFrame):
        df = _get_intrinsic_centroids_for_shapes(e, return_area)
    elif isinstance(e, DaskDataFrame):
        df = _get_intrinsic_centroids_for_points(e, return_area)
    else:
        raise ValueError(f"The object type {type(e)} is not supported.")
    coords = sorted(c for c in df.columns if c != "area")
    return df[coords + (["area"] if return_area else [])]


def _resolve_annotating_table(sdata: SpatialData, element_name: str, table_name: str | None) -> str:
    """Resolve the single table that annotates ``element_name`` (where centroids are written)."""
    from spatialdata._core.query.relational_query import get_element_annotators

    if table_name is not None:
        if table_name not in sdata.tables:
            raise KeyError(f"Table {table_name!r} not found in `sdata.tables`.")
        return table_name
    annotators = sorted(get_element_annotators(sdata, element_name))
    if not annotators:
        raise ValueError(
            f"Element {element_name!r} has no annotating table to write centroids into. Use `get_centroids()` to "
            f"get the centroids as a Points element instead, or annotate the element with a table first."
        )
    if len(annotators) > 1:
        raise ValueError(
            f"Element {element_name!r} is annotated by multiple tables ({', '.join(annotators)}); "
            f"pass `table_name=` to choose one."
        )
    return annotators[0]


def _put_centroids_into_table(table: AnnData, element_name: str, centroids: pd.DataFrame, overwrite: bool) -> None:
    """Write ``centroids`` into ``obsm["spatial"]`` (and an ``area`` column into ``obs["area"]``) at the element's rows.

    ``centroids`` is indexed by instance id, with coordinate columns ``x, y[, z]`` and optionally ``area``. Only the
    table rows annotating ``element_name`` are touched (a table may annotate several elements); instances annotated
    but absent from the element are written as NaN, and rows annotating other elements are preserved. If the target
    already holds values for the element's rows, the write is refused unless ``overwrite=True``. All checks run before
    anything is written, so a refused write leaves the table unchanged.
    """
    if not centroids.index.is_unique:
        raise ValueError(f"Cannot write centroids for {element_name!r}: its instance index has duplicate values.")
    _, region_key, instance_key = get_table_keys(table)
    mask = (table.obs[region_key].astype(str) == str(element_name)).to_numpy()
    if not mask.any():
        raise ValueError(f"The table does not annotate element {element_name!r} (no matching rows).")

    # Map each annotated instance to its centroid row (-1 where absent -> NaN). A *total* miss means the
    # instance ids never align with the element index (mismatched dtype, or no shared instances).
    keys = table.obs[instance_key].to_numpy()[mask]
    idx = centroids.index.get_indexer(pd.Index(keys))
    if (idx == -1).all():
        raise ValueError(
            f"No instance id annotating {element_name!r} is present in the element; check the table's "
            f"`{instance_key}` values and dtype."
        )
    hit = idx != -1

    def _scatter(values: np.ndarray) -> np.ndarray:
        """Gather ``values`` (ordered like ``centroids``) onto the masked rows, NaN where absent."""
        out = np.full((len(idx), *values.shape[1:]), np.nan)
        out[hit] = values[idx[hit]]
        return out

    coords = centroids.drop(columns="area", errors="ignore")
    ndim = coords.shape[1]
    spatial = np.full((table.n_obs, ndim), np.nan)
    existing = table.obsm.get(TableModel.SPATIAL_KEY)
    if existing is not None:
        existing = np.asarray(existing)
        if existing.shape != (table.n_obs, ndim):
            raise ValueError(
                f"Existing obsm['{TableModel.SPATIAL_KEY}'] {existing.shape} is incompatible with {ndim}-D centroids "
                f"for {element_name!r}; refusing to overwrite other regions."
            )
        spatial = existing.astype(float, copy=True)  # preserve other regions' coordinates
        if not overwrite and np.isfinite(spatial[mask]).any():
            raise ValueError(
                f"obsm['{TableModel.SPATIAL_KEY}'] already holds coordinates for rows annotating {element_name!r}; "
                f"pass overwrite=True to replace them."
            )
    spatial[mask] = _scatter(coords.to_numpy(dtype=float))

    area: np.ndarray | None = None
    if "area" in centroids.columns:
        area = np.full(table.n_obs, np.nan)
        if TableModel.AREA_KEY in table.obs:
            existing_area = table.obs[TableModel.AREA_KEY]
            if not pd.api.types.is_numeric_dtype(existing_area):
                raise ValueError(
                    f"obs['{TableModel.AREA_KEY}'] exists but is not numeric ({existing_area.dtype}); rename or drop "
                    f"it first."
                )
            area = np.array(existing_area, dtype=float)  # preserve other regions' values (copy)
            if not overwrite and np.isfinite(area[mask]).any():
                raise ValueError(
                    f"obs['{TableModel.AREA_KEY}'] already holds values for rows annotating {element_name!r} "
                    f"(this may be an unrelated column); pass overwrite=True to replace them."
                )
        area[mask] = _scatter(centroids["area"].to_numpy(dtype=float))

    table.obsm[TableModel.SPATIAL_KEY] = spatial
    if area is not None:
        table.obs[TableModel.AREA_KEY] = area


def add_centroids_to_table(
    sdata: SpatialData,
    element_name: str,
    *,
    table_name: str | None = None,
    return_area: bool = False,
    inplace: bool = True,
    overwrite: bool = False,
) -> AnnData | None:
    """
    Write the centroids of an element into the table that annotates it.

    The centroids go into ``obsm["spatial"]`` (columns ``x, y[, z]``) and, with ``return_area=True``, the sizes into
    ``obs["area"]``, at the rows annotating ``element_name``. Rows annotating other elements are left as they are;
    annotated instances that are absent from the element get NaN.

    The coordinates and sizes are **intrinsic** (untransformed): a table can annotate several elements, each with its
    own transformations, and the region column records which element each row belongs to. Use
    :func:`get_centroids` to get the centroids in a coordinate system instead.

    Parameters
    ----------
    sdata
        The SpatialData object containing the element and the table.
    element_name
        The name of the element (points, shapes or labels) to compute the centroids of.
    table_name
        The table to write into. If None, the single table annotating ``element_name`` is used.
    return_area
        If True, also write the size of each instance, measured as in :func:`get_centroids` but in intrinsic units.
    inplace
        If True, modify the table in ``sdata`` and return None. If False, leave ``sdata`` unchanged and return a
        modified copy of the table.
    overwrite
        If False, refuse to replace values that ``obsm["spatial"]`` or ``obs["area"]`` already hold for the rows of
        ``element_name``.

    Returns
    -------
    None if ``inplace`` is True, otherwise the modified copy of the table.
    """
    element = sdata[element_name]
    if isinstance(element, AnnData):
        raise ValueError(f"{element_name!r} is a table; centroids need a Labels, Shapes or Points element.")
    table_name = _resolve_annotating_table(sdata, element_name, table_name)
    centroids = _get_intrinsic_centroids(element, return_area)
    table = sdata.tables[table_name] if inplace else sdata.tables[table_name].copy()
    _put_centroids_into_table(table, element_name, centroids, overwrite)
    return None if inplace else table


##
