from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest
from anndata import AnnData
from numpy.random import default_rng
from xarray import DataTree

from spatialdata._core.centroids import _put_centroids_into_table, add_centroids_to_table, get_centroids
from spatialdata._core.query.relational_query import get_element_instances
from spatialdata.models import Labels2DModel, Labels3DModel, PointsModel, TableModel, get_axes_names
from spatialdata.transformations import Affine, Identity, Scale, get_transformation, set_transformation

RNG = default_rng(42)


def _get_affine() -> Affine:
    theta: float = math.pi / 18
    k = 10.0
    return Affine(
        [
            [2 * math.cos(theta), 2 * math.sin(-theta), -1000 / k],
            [2 * math.sin(theta), 2 * math.cos(theta), 300 / k],
            [0, 0, 1],
        ],
        input_axes=("x", "y"),
        output_axes=("x", "y"),
    )


affine = _get_affine()


@pytest.mark.parametrize("coordinate_system", ["global", "aligned"])
@pytest.mark.parametrize("is_3d", [False, True])
def test_get_centroids_points(points, coordinate_system: str, is_3d: bool):
    element = points["points_0"].compute()
    element.index = np.arange(len(element)) + 10
    element = PointsModel.parse(element)

    # by default, the coordinate system is global and the points are 2D; let's modify the points as instructed by the
    # test arguments
    if coordinate_system == "aligned":
        set_transformation(element, transformation=affine, to_coordinate_system=coordinate_system)
    if is_3d:
        element["z"] = element["x"]

    axes = get_axes_names(element)
    centroids = get_centroids(element, coordinate_system=coordinate_system)

    # the axes of the centroids should be the same as the axes of the element
    assert centroids.columns.tolist() == list(axes)

    # check the index is preserved
    assert np.array_equal(centroids.index.compute().values, element.index.compute().values)

    # the centroids should not contain extra columns
    assert "genes" in element.columns and "genes" not in centroids.columns

    # the centroids transformation to the target coordinate system should be an Identity because the transformation has
    # already been applied
    assert get_transformation(centroids, to_coordinate_system=coordinate_system) == Identity()

    # let's check the values
    if coordinate_system == "global":
        assert np.array_equal(centroids.compute().values, element[list(axes)].compute().values)
    else:
        matrix = affine.to_affine_matrix(input_axes=axes, output_axes=axes)
        centroids_untransformed = element[list(axes)].compute().values
        n = len(axes)
        centroids_transformed = np.dot(centroids_untransformed, matrix[:n, :n].T) + matrix[:n, n]
        assert np.allclose(centroids.compute().values, centroids_transformed)


@pytest.mark.parametrize("coordinate_system", ["global", "aligned"])
@pytest.mark.parametrize("shapes_name", ["circles", "poly", "multipoly"])
def test_get_centroids_shapes(shapes, coordinate_system: str, shapes_name: str):
    element = shapes[shapes_name]
    element.index = np.arange(len(element)) + 10

    if coordinate_system == "aligned":
        set_transformation(element, transformation=affine, to_coordinate_system=coordinate_system)
    centroids = get_centroids(element, coordinate_system=coordinate_system)

    assert np.array_equal(centroids.index.compute().values, element.index.values)

    if shapes_name == "circles":
        xy = element.geometry.get_coordinates().values
    else:
        assert shapes_name in ["poly", "multipoly"]
        xy = element.geometry.centroid.get_coordinates().values

    if coordinate_system == "global":
        assert np.array_equal(centroids.compute().values, xy)
    else:
        matrix = affine.to_affine_matrix(input_axes=("x", "y"), output_axes=("x", "y"))
        centroids_transformed = np.dot(xy, matrix[:2, :2].T) + matrix[:2, 2]
        assert np.allclose(centroids.compute().values, centroids_transformed)


@pytest.mark.parametrize("coordinate_system", ["global", "aligned"])
@pytest.mark.parametrize("is_multiscale", [False, True])
@pytest.mark.parametrize("is_3d", [False, True])
@pytest.mark.parametrize("return_background", [False, True])
def test_get_centroids_labels(
    labels, coordinate_system: str, is_multiscale: bool, is_3d: bool, return_background: bool
):
    scale_factors = [2] if is_multiscale else None
    if is_3d:
        model = Labels3DModel
        array = np.array(
            [
                [
                    [0, 0, 10, 10],
                    [0, 0, 10, 10],
                ],
                [
                    [20, 20, 10, 10],
                    [20, 20, 10, 10],
                ],
            ]
        )
        expected_centroids = pd.DataFrame(
            {
                "x": [1, 3, 1],
                "y": [1, 1.0, 1],
                "z": [0.5, 1, 1.5],
            },
            index=[0, 1, 2],
        )
        if not return_background:
            expected_centroids = expected_centroids.drop(index=0)
    else:
        array = np.array(
            [
                [10, 10, 10, 10],
                [20, 20, 20, 20],
                [20, 20, 20, 20],
                [20, 20, 20, 20],
            ]
        )
        model = Labels2DModel
        expected_centroids = pd.DataFrame(
            {
                "x": [2, 2],
                "y": [0.5, 2.5],
            },
            index=[1, 2],
        )
    element = model.parse(array, scale_factors=scale_factors)

    if coordinate_system == "aligned":
        set_transformation(element, transformation=affine, to_coordinate_system=coordinate_system)
    centroids = get_centroids(element, coordinate_system=coordinate_system, return_background=return_background)

    labels_indices = get_element_instances(element, return_background=return_background)
    assert np.array_equal(centroids.index.compute().values, labels_indices)

    if not return_background:
        assert not (centroids.index == 0).any()

    if coordinate_system == "global":
        assert np.array_equal(centroids.compute().values, expected_centroids.values)
    else:
        axes = get_axes_names(element)
        n = len(axes)
        # the axes from the labels have 'x' last, but we want it first to manually transform the points, so we sort
        matrix = affine.to_affine_matrix(input_axes=sorted(axes), output_axes=sorted(axes))
        centroids_transformed = np.dot(expected_centroids.values, matrix[:n, :n].T) + matrix[:n, n]
        assert np.allclose(centroids.compute().values, centroids_transformed)


def test_get_centroids_labels_area(labels):
    # area for labels is the per-label pixel count; it rides along as a feature column of the Points.
    element = labels["labels2d"]
    centroids = get_centroids(element, return_area=True)
    assert "area" in centroids.columns
    ids, counts = np.unique(np.asarray(element.data), return_counts=True)
    expected = dict(zip(ids, counts, strict=True))
    got = centroids[["area"]].compute()["area"]
    assert not (got.index == 0).any()  # background dropped
    for label_id, area in got.items():
        assert area == expected[label_id]


def test_get_centroids_shapes_area_circles(shapes):
    element = shapes["circles"]
    centroids = get_centroids(element, return_area=True)
    expected = np.pi * np.asarray(element["radius"], dtype=float) ** 2
    assert np.allclose(centroids["area"].compute().to_numpy(), expected)


@pytest.mark.parametrize("shapes_name", ["poly", "multipoly"])
def test_get_centroids_shapes_area_polygons(shapes, shapes_name: str):
    element = shapes[shapes_name]
    centroids = get_centroids(element, return_area=True)
    assert np.allclose(centroids["area"].compute().to_numpy(), element.geometry.area.to_numpy())


def test_get_centroids_points_area_raises(points):
    with pytest.raises(ValueError, match="not supported for points"):
        get_centroids(points["points_0"], return_area=True)


def _label_counts(element) -> dict[int, int]:
    """Pixel/voxel count per label id, measured on the full-resolution level."""
    data = next(iter(element.children["scale0"].data_vars.values())) if isinstance(element, DataTree) else element
    ids, counts = np.unique(data.to_numpy(), return_counts=True)
    return dict(zip(ids.tolist(), counts.tolist(), strict=True))


def test_get_centroids_labels_area_multiscale(labels):
    # multiscale labels are measured on the full-resolution scale0 level
    element = labels["labels2d_multiscale"]
    area = get_centroids(element, return_area=True)["area"].compute()
    expected = _label_counts(element)
    assert area.to_dict() == {k: float(v) for k, v in expected.items() if k != 0}


def test_get_centroids_labels3d_area(labels):
    # for 3D labels, `area` is the voxel count (a volume), as in skimage's regionprops
    element = labels["labels3d_numpy"]
    centroids = get_centroids(element, return_area=True, return_background=True).compute()
    assert centroids["area"].to_dict() == {k: float(v) for k, v in _label_counts(element).items()}


def test_get_centroids_area_non_identity_transform_raises(full_sdata):
    # get_centroids transforms the centroids but the area is intrinsic (untransformed); requesting
    # both under a non-identity transform must raise rather than mix coordinate systems.
    set_transformation(full_sdata["labels2d"], affine, "aligned")
    with pytest.raises(ValueError, match="requires intrinsic coordinates"):
        get_centroids(full_sdata["labels2d"], coordinate_system="aligned", return_area=True)


def test_get_centroids_invalid_element(images):
    # cannot compute centroids for images
    with pytest.raises(ValueError, match="Centroids are not supported for elements modeled by Image2DModel"):
        get_centroids(images["image2d"])

    # cannot compute centroids for tables
    N = 10
    adata = TableModel.parse(
        AnnData(X=RNG.random((N, N)), obs={"region": pd.Categorical(["dummy"] * N), "instance_id": np.arange(N)}),
        region="dummy",
        region_key="region",
        instance_key="instance_id",
    )
    with pytest.raises(ValueError, match=r"The object type <class 'anndata.*AnnData'> is not supported"):
        get_centroids(adata)


def test_get_centroids_invalid_coordinate_system(points):
    with pytest.raises(ValueError, match="No transformation to coordinate system"):
        get_centroids(points["points_0"], coordinate_system="invalid")


def test_get_centroids_numerically_identity_transform_is_intrinsic(full_sdata):
    # a mathematically-identity transform that is not the `Identity` class (e.g. Scale([1, 1])) must be
    # accepted as intrinsic for return_area.
    set_transformation(full_sdata["labels2d"], Scale([1.0, 1.0], axes=("x", "y")), "noop")
    # return_area on the Points path must not raise "requires intrinsic coordinates"
    out = get_centroids(full_sdata["labels2d"], coordinate_system="noop", return_area=True)
    assert "area" in out.columns


def test_get_centroids_points_area_non_identity_reports_points_not_intrinsic(points):
    # a Points element with a non-identity transform + return_area must report the points-specific error,
    # not the misleading "requires intrinsic coordinates" message.
    set_transformation(points["points_0"], affine, "aligned")
    with pytest.raises(ValueError, match="not supported for points"):
        get_centroids(points["points_0"], coordinate_system="aligned", return_area=True)


def _assert_obsm_matches_points(table: AnnData, pts: pd.DataFrame) -> None:
    # written obsm["spatial"] must match the element-level Points centroids on shared (non-background) ids.
    inst = table.obs["instance_id"].to_numpy()
    written = pd.DataFrame(table.obsm["spatial"], index=inst, columns=["x", "y"])
    common = pts.index.intersection(written.index[inst != 0])
    assert len(common) > 0
    assert np.allclose(written.loc[common].to_numpy(), pts.loc[common][["x", "y"]].to_numpy())


def test_add_centroids_to_table(full_sdata):
    # `table` has 100 rows annotating `labels2d`; the `instance_id` column holds the label ids each row
    # refers to (the value at a given row position is the label, not the row's position). The background
    # label 0 has no centroid, so its row stays NaN.
    table = full_sdata["table"]
    assert "spatial" not in table.obsm
    out = add_centroids_to_table(full_sdata, "labels2d", return_area=True)
    assert out is None  # inplace=True (default) mutates the table and returns nothing

    table = full_sdata["table"]
    assert table.obsm["spatial"].shape == (table.n_obs, 2)
    assert "area" in table.obs

    inst = table.obs["instance_id"].to_numpy()
    finite = np.isfinite(table.obsm["spatial"]).all(axis=1)
    assert finite[inst != 0].all()  # every non-background label got a centroid
    assert not finite[inst == 0].any()  # background row stays NaN

    # coordinates must match the element-level Points (global transform is the identity here)
    pts = get_centroids(full_sdata["labels2d"], coordinate_system="global").compute()
    _assert_obsm_matches_points(table, pts)

    # area must equal the pixel counts of the corresponding labels
    count_of = _label_counts(full_sdata["labels2d"])
    area = table.obs["area"].to_numpy()
    for row, label_id in enumerate(inst):
        if label_id != 0:
            assert area[row] == count_of[label_id]


def test_add_centroids_to_table_is_intrinsic(full_sdata):
    # the table stores intrinsic coordinates and sizes, whatever the element's transformations
    element = full_sdata["labels2d"]
    set_transformation(element, affine, "global")
    set_transformation(element, Identity(), "intrinsic")
    add_centroids_to_table(full_sdata, "labels2d", return_area=True)
    pts = get_centroids(element, coordinate_system="intrinsic", return_area=True).compute()
    table = full_sdata["table"]
    _assert_obsm_matches_points(table, pts)
    inst = table.obs["instance_id"].to_numpy()
    assert np.allclose(table.obs["area"].to_numpy()[inst != 0], pts.loc[inst[inst != 0], "area"].to_numpy())


def test_add_centroids_to_table_inplace_false_returns_copy(full_sdata):
    # inplace=False copies only the target table, writes into the copy, and leaves the sdata untouched.
    out = add_centroids_to_table(full_sdata, "labels2d", return_area=True, inplace=False)
    assert isinstance(out, AnnData)
    assert out is not full_sdata["table"]
    assert "spatial" in out.obsm and "area" in out.obs
    assert "spatial" not in full_sdata["table"].obsm  # original table not modified


def test_add_centroids_to_table_instance_key_mismatch_raises(full_sdata):
    # instance ids stored with a dtype that doesn't match the element's integer labels must fail
    # loudly instead of silently writing NaN coordinates into obsm["spatial"].
    table = full_sdata["table"]
    table.obs["instance_id"] = table.obs["instance_id"].astype(str)
    with pytest.raises(ValueError, match="No instance id annotating"):
        add_centroids_to_table(full_sdata, "labels2d")


def test_add_centroids_to_table_refuses_dim_mismatch(full_sdata):
    # This mismatch arises when one table annotates multiple elements of different dimensionality: e.g. a
    # 3D element already wrote width-3 coordinates into obsm["spatial"], and we now write 2D centroids
    # for `labels2d`. The existing width-3 array (simulated here) must not be silently overwritten, as
    # that would wipe the other element's coordinates sharing the table.
    table = full_sdata["table"]
    table.obsm["spatial"] = np.zeros((table.n_obs, 3))
    with pytest.raises(ValueError, match="refusing to overwrite"):
        add_centroids_to_table(full_sdata, "labels2d")


def test_add_centroids_to_table_no_table_raises(full_sdata):
    # points_0 is not annotated by any table -> the error points the user to get_centroids
    with pytest.raises(ValueError, match="get_centroids"):
        add_centroids_to_table(full_sdata, "points_0")


def test_add_centroids_to_table_overwrite_guard(full_sdata):
    # a pre-existing obs['area'] (possibly unrelated) must not be silently clobbered; overwrite=True allows it.
    table = full_sdata["table"]
    table.obs["area"] = np.arange(table.n_obs, dtype=float)
    with pytest.raises(ValueError, match="overwrite=True"):
        add_centroids_to_table(full_sdata, "labels2d", return_area=True)
    # with overwrite=True it proceeds and writes the pixel-count area
    add_centroids_to_table(full_sdata, "labels2d", return_area=True, overwrite=True)
    assert "spatial" in table.obsm

    # obsm["spatial"] is likewise guarded once populated
    with pytest.raises(ValueError, match="overwrite=True"):
        add_centroids_to_table(full_sdata, "labels2d")


def _two_region_table(area: list) -> AnnData:
    obs = pd.DataFrame(
        {"region": pd.Categorical(["a", "a", "b", "b"]), "instance_id": [1, 2, 1, 2], "area": area},
        index=[str(i) for i in range(4)],
    )
    return TableModel.parse(AnnData(obs=obs), region=["a", "b"], region_key="region", instance_key="instance_id")


def test_put_centroids_into_table_keeps_other_regions():
    # only the rows of the given element are written; the other region's values stay as they are
    table = _two_region_table([np.nan, np.nan, 5.0, 6.0])
    table.obsm["spatial"] = np.array([[np.nan, np.nan], [np.nan, np.nan], [7.0, 8.0], [9.0, 10.0]])
    centroids = pd.DataFrame({"x": [1.0, 2.0], "y": [3.0, 4.0], "area": [11.0, 12.0]}, index=[1, 2])
    _put_centroids_into_table(table, "a", centroids, overwrite=False)
    assert np.array_equal(table.obsm["spatial"], [[1.0, 3.0], [2.0, 4.0], [7.0, 8.0], [9.0, 10.0]])
    assert table.obs["area"].tolist() == [11.0, 12.0, 5.0, 6.0]
