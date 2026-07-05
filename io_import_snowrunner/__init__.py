# SPDX-License-Identifier: MIT
"""
Blender importer for SnowRunner / MudRunner-style "CombineXMesh" binary mesh files
(the format described by the accompanying ImHex pattern).

Imports:
  * full node hierarchy as an armature (one bone per node)
  * meshes (visual + optional collision), with UVs (2 layers), custom split
    normals, per-submesh material assignment
  * skinning: vertex groups built from per-vertex bone links/weights, with
    per-submesh bone palettes; unskinned meshes are rigidly bound to their node
  * material names + albedo/tint texture paths parsed from the embedded XML
    header (textures are loaded from an optional user-supplied folder)

Coordinate conversion: the game data is right-handed, Y-up; it is rotated
+90 degrees about X into Blender's Z-up space (x, y, z) -> (x, -z, y).
"""

bl_info = {
    "name": "Snowrunner Model Importer",
    "author": "Brooen",
    "version": (2, 0, 0),
    "blender": (5, 1, 0),
    "location": "File > Import > Import Snowrunner Model ([meshes])",
    "category": "Import-Export",
}

import os
import re
import struct

try:
    import bpy
    from bpy_extras.io_utils import ImportHelper
    from mathutils import Matrix, Vector
    HAS_BPY = True
except ImportError:  # allows the parser half to be tested outside Blender
    HAS_BPY = False


# ---------------------------------------------------------------------------
# Binary parser
# ---------------------------------------------------------------------------

ITEM_POSITION = 0x0000
ITEM_UV       = 0x0005
ITEM_NORMAL   = 0x0105
ITEM_UNK205   = 0x0205
ITEM_UNK305   = 0x0305
ITEM_WEIGHT   = 0x0405
ITEM_LINK     = 0x0505
ITEM_UV2      = 0x0605  # two floats in [0,1] -> second UV set

ITEM_SIZES = {
    ITEM_POSITION: 12, ITEM_UV: 8, ITEM_NORMAL: 4, ITEM_UNK205: 4,
    ITEM_UNK305: 4, ITEM_WEIGHT: 4, ITEM_LINK: 4, ITEM_UV2: 8,
}


class MeshFormatError(Exception):
    pass


class Reader:
    __slots__ = ("d", "o")

    def __init__(self, data):
        self.d = data
        self.o = 0

    def tell(self):
        return self.o

    def seek(self, o):
        self.o = o

    def read(self, n):
        b = self.d[self.o:self.o + n]
        if len(b) != n:
            raise MeshFormatError(
                "Unexpected end of file (wanted %d bytes at 0x%X)" % (n, self.o))
        self.o += n
        return b

    def s32(self):
        return struct.unpack_from("<i", self.d, self._adv(4))[0]

    def u32(self):
        return struct.unpack_from("<I", self.d, self._adv(4))[0]

    def s16(self):
        return struct.unpack_from("<h", self.d, self._adv(2))[0]

    def u16(self):
        return struct.unpack_from("<H", self.d, self._adv(2))[0]

    def f32(self):
        return struct.unpack_from("<f", self.d, self._adv(4))[0]

    def f32s(self, n):
        return struct.unpack_from("<%df" % n, self.d, self._adv(4 * n))

    def _adv(self, n):
        o = self.o
        if o + n > len(self.d):
            raise MeshFormatError(
                "Unexpected end of file (wanted %d bytes at 0x%X)" % (n, o))
        self.o = o + n
        return o

    def name(self):
        ln = self.s32()
        if ln <= 0 or ln > 4096:
            raise MeshFormatError("Bad string length %d at 0x%X" % (ln, self.o - 4))
        s = self.read(ln - 1).decode("cp1251", errors="replace")
        self.read(1)  # trailing NUL
        return s

    def matrix_rows(self):
        """4 rows of 4 floats: dirX, dirY, dirZ, pos (row-vector convention)."""
        return [list(self.f32s(4)) for _ in range(4)]


def _parse_vertex_type(r):
    count = r.s32()
    if count < 0 or count > 64:
        raise MeshFormatError("Bad vertexType count %d at 0x%X" % (count, r.tell() - 4))
    defs = []
    for _ in range(count):
        r.s16()               # unknown1
        off = r.s16()
        dt = r.u16()          # dataType
        it = r.u16()          # itemType
        if it not in ITEM_SIZES:
            raise MeshFormatError("Unknown vertex item type 0x%04X" % it)
        defs.append((off, dt, it))
    r.s32()  # flag1
    r.s32()  # flag2
    return defs


def _parse_vertices(r, defs, count):
    """Returns dict of arrays: positions, normals, uv, uv2, weights, links."""
    positions = []
    normals = [] if any(d[2] == ITEM_NORMAL for d in defs) else None
    uv = [] if any(d[2] == ITEM_UV for d in defs) else None
    uv2 = [] if any(d[2] == ITEM_UV2 for d in defs) else None
    weights = [] if any(d[2] == ITEM_WEIGHT for d in defs) else None
    links = [] if any(d[2] == ITEM_LINK for d in defs) else None

    for _ in range(count):
        for _, _, it in defs:
            if it == ITEM_POSITION:
                positions.append(r.f32s(3))
            elif it == ITEM_UV:
                uv.append(r.f32s(2))
            elif it == ITEM_NORMAL:
                b = r.read(4)
                normals.append(tuple(
                    (float(c) - 128.0) / (127.0 if c >= 128 else 128.0)
                    for c in b[:3]))
            elif it == ITEM_UV2:
                uv2.append(r.f32s(2))
            elif it == ITEM_WEIGHT:
                b = r.read(4)
                weights.append((b[0] / 255.0, b[1] / 255.0, b[2] / 255.0, b[3] / 255.0))
            elif it == ITEM_LINK:
                links.append(tuple(r.read(4)))
            else:
                r.read(ITEM_SIZES[it])

    return {
        "positions": positions, "normals": normals, "uv": uv, "uv2": uv2,
        "weights": weights, "links": links,
    }


def _parse_submesh_header(r):
    return {
        "materialIndex": r.u32(),
        "triangleOffset": r.u32(),
        "triangleCount": r.u32(),
        "vertexOffset": r.u32(),
        "vertexCount": r.u32(),
    }


def _parse_mesh(r, link_in_count):
    m = {}
    m["vertexCount"] = r.s32()
    m["triangleCount"] = r.s32()
    m["name"] = r.name()
    r.s32()                                # UNKNOWN1
    material_count = r.s32()
    r.s32()                                # UNKNOWN2
    if not (0 <= material_count <= 4096):
        raise MeshFormatError("Bad material count %d" % material_count)
    m["materials"] = [r.name() for _ in range(material_count)]

    link_out = r.s32()
    if not (0 <= link_out <= 4096):
        raise MeshFormatError("Bad linkOutCount %d" % link_out)
    m["linkOutCount"] = link_out
    m["linkMatrices"] = [r.matrix_rows() for _ in range(link_out)]
    r.s16()                                # indexOfType

    if link_out == 0:
        r.f32s(6)                          # limits
        submesh_count = r.s32()
        m["submeshes"] = [_parse_submesh_header(r) for _ in range(submesh_count)]
        for s in m["submeshes"]:
            s["indices"] = None
        m["linkedNodes"] = []
        defs = _parse_vertex_type(r)
        m["vertexDefs"] = defs
        m["vdata"] = _parse_vertices(r, defs, m["vertexCount"])
        m["triangles"] = list(struct.iter_unpack(
            "<3H", r.read(6 * m["triangleCount"])))
        if link_in_count != 0:
            r.s16()                        # extraDataIndex
    else:
        r.s16()                            # UNKNOWN3
        submesh_count = r.s32()
        index_counts = [r.s32() for _ in range(submesh_count)]
        subs = []
        for i in range(submesh_count):
            s = _parse_submesh_header(r)
            s["indices"] = [r.s32() for _ in range(index_counts[i])]
            subs.append(s)
        m["submeshes"] = subs
        m["linkedNodes"] = [r.s16() for _ in range(link_out)]
        r.f32s(6)                          # limits
        r.read(4 * 6)                      # index[2] + sub tri/vert offsets/counts
        defs = _parse_vertex_type(r)
        m["vertexDefs"] = defs
        m["vdata"] = _parse_vertices(r, defs, m["vertexCount"])
        m["triangles"] = list(struct.iter_unpack(
            "<3H", r.read(6 * m["triangleCount"])))
        r.s16()                            # extraDataIndex

    flag = r.s16()
    if 4 <= flag <= 17:
        r.read(1)
        count = r.s16()
        r.read(1)
        r.read(count + 16)                 # opaque blob
        flag2 = r.s16()
        if flag2 > 100:
            r.matrix_rows()
    if flag > 100:
        r.matrix_rows()
    return m


def _parse_node(r):
    n = {}
    n["parentId"] = r.s16()
    n["id"] = r.s16()
    n["linkInCount"] = r.s16()
    r.s16()                                # SPACE1
    n["name"] = r.name()
    n["matrix"] = r.matrix_rows()
    pos = r.tell()
    vertex_count = r.s32()
    r.seek(pos)
    if vertex_count != 0:
        n["mesh"] = _parse_mesh(r, n["linkInCount"])
    else:
        r.read(4)
        n["mesh"] = None
    return n


def parse_combinexmesh(filepath):
    with open(filepath, "rb") as fh:
        data = fh.read()
    r = Reader(data)

    xml_length = r.s32()
    if not (2 < xml_length < len(data)):
        raise MeshFormatError("Not a CombineXMesh file (bad XML header length)")
    xml = r.read(xml_length - 2).decode("cp1251", errors="replace")
    if "<" not in xml[:64]:
        raise MeshFormatError("Not a CombineXMesh file (missing XML header)")
    r.read(6)                              # 3 x s16 spacers

    node_count = r.s32()
    r.f32s(6)                              # global limits
    r.s32()                                # meshCount
    if not (0 < node_count <= 65535):
        raise MeshFormatError("Bad node count %d" % node_count)

    nodes = []
    for i in range(node_count):
        try:
            nodes.append(_parse_node(r))
        except MeshFormatError as e:
            raise MeshFormatError("Failed parsing node %d/%d: %s" % (i, node_count, e))
    # shaftData / extraMesh sections follow; not needed for import.
    return {"xml": xml, "nodes": nodes}


def parse_xml_materials(xml):
    """Extract material property dicts from SnowRunner XML.

    Handles both full <Material .../> definitions (material .xml files) and
    the <MaterialOverride .../> entries embedded in mesh headers.  Returns
    {material_name: {attr: value, ...}}, where the name is TargetMaterialName
    for overrides and Name for plain materials.
    """
    result = {}
    for tag, block in re.findall(
            r"<(MaterialOverride|Material)\b(.*?)/>", xml, re.S):
        attrs = dict(re.findall(r'(\w+)\s*=\s*"([^"]*)"', block))
        name = attrs.get("TargetMaterialName") if tag == "MaterialOverride" \
            else attrs.get("Name")
        if name:
            merged = result.setdefault(name, {})
            merged.update(attrs)
    return result


# ---------------------------------------------------------------------------
# Blender import
# ---------------------------------------------------------------------------

if HAS_BPY:

    # Game space is right-handed Y-up; rotate +90deg about X into Blender's
    # Z-up space and mirror across X (the game data is left/right flipped
    # relative to Blender): (x, y, z) -> (-x, -z, y).
    CONV = Matrix(((-1, 0, 0, 0),
                   (0, 0, -1, 0),
                   (0, 1, 0, 0),
                   (0, 0, 0, 1)))

    def _rows_to_matrix(rows):
        """File matrices are row-vector convention (v' = v @ M); transpose to
        Blender's column-vector convention."""
        return Matrix(rows).transposed()

    def _compute_world_matrices(nodes):
        """World matrix per node index, in game space (column convention)."""
        by_id = {}
        for i, n in enumerate(nodes):
            by_id.setdefault(n["id"], i)
        worlds = [None] * len(nodes)

        def world(i):
            if worlds[i] is not None:
                return worlds[i]
            n = nodes[i]
            local = _rows_to_matrix(n["matrix"])
            pid = n["parentId"]
            if pid >= 0 and pid in by_id and by_id[pid] != i:
                m = world(by_id[pid]) @ local
            else:
                m = local
            worlds[i] = m
            return m

        for i in range(len(nodes)):
            world(i)
        return worlds, by_id

    def _unique_bone_names(nodes):
        names = {}
        used = set()
        for i, n in enumerate(nodes):
            base = n["name"] or ("node_%d" % n["id"])
            name = base
            k = 1
            while name in used:
                name = "%s.%03d" % (base, k)
                k += 1
            used.add(name)
            names[i] = name
        return names

    # ------------------------------------------------------------------
    # Materials (SnowRunner node-group shader)
    # ------------------------------------------------------------------

    SHADER_GROUP_NAME = "Snowrunner Shader"
    TEXTURE_EXTS = (".dds", ".tga", ".png", ".jpg", ".tif")
    NONCOLOR_KEYS = ("NormalMap", "ShadingMap")

    def ensure_shader_group():
        """Return the SnowRunner shader node group, appending it from the
        shader.blend bundled with the addon on first use."""
        group = bpy.data.node_groups.get(SHADER_GROUP_NAME)
        if group is not None:
            return group
        blend = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             "shader.blend")
        if not os.path.isfile(blend):
            return None
        try:
            with bpy.data.libraries.load(blend, link=False) as (src, dst):
                if SHADER_GROUP_NAME in src.node_groups:
                    dst.node_groups = [SHADER_GROUP_NAME]
        except Exception as e:
            print("CombineXMesh: could not load shader.blend: %r" % e)
            return None
        return bpy.data.node_groups.get(SHADER_GROUP_NAME)

    def _set_alpha_blend(mat, clip):
        # Blender < 4.2 (EEVEE Legacy)
        if hasattr(mat, "blend_method"):
            try:
                mat.blend_method = "CLIP" if clip else "BLEND"
            except TypeError:  # 4.2+: enum reduced, or property gone
                pass
        if clip and hasattr(mat, "alpha_threshold"):
            mat.alpha_threshold = 0.5
        # Blender 4.2+ (EEVEE Next)
        if hasattr(mat, "surface_render_method") and not clip:
            mat.surface_render_method = "BLENDED"

    class MaterialBuilder:
        """Creates/updates Blender materials from SnowRunner XML attributes.

        Indexes the textures folder once (instead of walking it per texture)
        and reuses materials across meshes within an import session.
        """

        def __init__(self, tex_dir, use_shader=True):
            self.tex_dir = tex_dir
            self.cache = {}
            self.file_index = self._index_textures(tex_dir)
            self.shader_group = ensure_shader_group() if use_shader else None

        @staticmethod
        def _index_textures(tex_dir):
            index = {}
            if tex_dir and os.path.isdir(tex_dir):
                for root, _dirs, files in os.walk(tex_dir):
                    for f in files:
                        index.setdefault(f.lower(), os.path.join(root, f))
            return index

        def resolve_texture(self, ref):
            """Map a reference like 'trucks/gor_by4__d.tga' to a file on disk.

            Tries: the exact relative path; the basename with common
            extensions (game .tga refs are usually stored as .dds); the
            path flattened with underscores (pak-extractor convention);
            finally a suffix match against the index."""
            if not ref or not self.file_index and not self.tex_dir:
                return None
            rel = ref.replace("\\", "/").strip("/")
            direct = os.path.join(self.tex_dir, rel)
            if os.path.isfile(direct):
                return direct

            stem = os.path.splitext(os.path.basename(rel))[0].lower()
            flat = os.path.splitext(rel)[0].replace("/", "_").lower()
            for base in (stem, flat):
                for ext in TEXTURE_EXTS:
                    hit = self.file_index.get(base + ext)
                    if hit:
                        return hit
            # last resort: any indexed file ending with the stem
            for ext in TEXTURE_EXTS:
                suffix = stem + ext
                for fname, path in self.file_index.items():
                    if fname.endswith(suffix):
                        return path
            return None

        def _load_image(self, path, key):
            img = bpy.data.images.load(path, check_existing=True)
            if any(k in key for k in NONCOLOR_KEYS):
                img.colorspace_settings.name = "Non-Color"
            else:
                img.alpha_mode = "CHANNEL_PACKED"
            return img

        def get(self, name, props=None):
            """Return (creating if needed) the material `name`, configured
            from SnowRunner attributes `props`."""
            if name in self.cache:
                return self.cache[name]
            props = props or {}
            mat = bpy.data.materials.get(name) or bpy.data.materials.new(name)
            self.cache[name] = mat
            try:
                self._build(mat, props)
            except Exception as e:
                print("CombineXMesh: material %r build failed: %r" % (name, e))
            return mat

        def _build(self, mat, props):
            mat.use_nodes = True
            nodes = mat.node_tree.nodes
            links = mat.node_tree.links
            nodes.clear()

            output = nodes.new("ShaderNodeOutputMaterial")
            output.location = (320, 0)

            if self.shader_group is not None:
                shader = nodes.new("ShaderNodeGroup")
                shader.node_tree = self.shader_group
                shader.label = SHADER_GROUP_NAME
                links.new(shader.outputs["BSDF"], output.inputs["Surface"])
            else:
                shader = nodes.new("ShaderNodeBsdfPrincipled")
                links.new(shader.outputs["BSDF"], output.inputs["Surface"])
            shader.location = (0, 0)

            wants_alpha = (props.get("Blending", "").lower() == "alpha"
                           or props.get("AlphaKill", "").lower() == "true")

            row = 0
            for key, value in props.items():
                if not key.endswith("Map") or not value:
                    continue
                path = self.resolve_texture(value)
                if not path:
                    print("CombineXMesh: texture not found: %s" % value)
                    continue
                tex = nodes.new("ShaderNodeTexImage")
                tex.image = self._load_image(path, key)
                tex.label = key
                tex.location = (-420, 300 - row * 300)
                row += 1

                if self.shader_group is not None:
                    if key in shader.inputs:
                        links.new(tex.outputs["Color"], shader.inputs[key])
                    if key == "AlbedoMap" and wants_alpha \
                            and "AlbedoMapAlpha" in shader.inputs:
                        links.new(tex.outputs["Alpha"],
                                  shader.inputs["AlbedoMapAlpha"])
                else:
                    # Principled fallback wiring
                    if key == "AlbedoMap":
                        links.new(tex.outputs["Color"],
                                  shader.inputs["Base Color"])
                        if wants_alpha:
                            links.new(tex.outputs["Alpha"],
                                      shader.inputs["Alpha"])
                    elif key == "NormalMap":
                        nm = nodes.new("ShaderNodeNormalMap")
                        nm.location = (-180, -300)
                        links.new(tex.outputs["Color"], nm.inputs["Color"])
                        links.new(nm.outputs["Normal"],
                                  shader.inputs["Normal"])

            if wants_alpha:
                _set_alpha_blend(
                    mat, clip=props.get("AlphaKill", "").lower() == "true")

    def _build_armature(context, base_name, nodes, worlds, bone_names, bone_size):
        arm_data = bpy.data.armatures.new(base_name + "_rig")
        arm_obj = bpy.data.objects.new(base_name + "_rig", arm_data)
        context.collection.objects.link(arm_obj)
        context.view_layer.objects.active = arm_obj
        arm_obj.select_set(True)
        bpy.ops.object.mode_set(mode="EDIT")
        try:
            ebones = []
            for i, n in enumerate(nodes):
                eb = arm_data.edit_bones.new(bone_names[i])
                m = CONV @ worlds[i]
                head = m.translation.copy()
                y_axis = Vector((m[0][1], m[1][1], m[2][1]))
                z_axis = Vector((m[0][2], m[1][2], m[2][2]))
                if y_axis.length < 1e-8:
                    y_axis = Vector((0, 1, 0))
                eb.head = head
                eb.tail = head + y_axis.normalized() * bone_size
                if z_axis.length > 1e-8:
                    eb.align_roll(z_axis.normalized())
                ebones.append(eb)
            by_id = {}
            for i, n in enumerate(nodes):
                by_id.setdefault(n["id"], i)
            for i, n in enumerate(nodes):
                pid = n["parentId"]
                if pid >= 0 and pid in by_id and by_id[pid] != i:
                    ebones[i].parent = ebones[by_id[pid]]
                    ebones[i].use_connect = False
        finally:
            bpy.ops.object.mode_set(mode="OBJECT")
        return arm_obj

    def _build_mesh_object(context, node, node_index, nodes, worlds, by_id,
                           bone_names, arm_obj, materials, overrides, options):
        m = node["mesh"]
        vdata = m["vdata"]
        positions = vdata["positions"]
        if not positions:
            return None

        skinned = m["linkOutCount"] > 0
        world = worlds[node_index]

        # Vertex positions -> Blender space.
        if skinned:
            # already in model space
            verts = [(CONV @ Vector(p)) for p in positions]
            normal_xform = CONV.to_3x3()
        else:
            xf = CONV @ world
            verts = [(xf @ Vector(p)) for p in positions]
            normal_xform = (CONV @ world).to_3x3()
            try:
                normal_xform = normal_xform.inverted().transposed()
            except ValueError:
                pass

        scale = options["scale"]
        if scale != 1.0:
            verts = [v * scale for v in verts]

        # CONV mirrors X (reflection), which inverts triangle winding;
        # reverse it so faces stay outward-facing.
        faces = [(t[0], t[2], t[1]) for t in m["triangles"]]

        mesh_name = m["name"] or node["name"] or "mesh"
        me = bpy.data.meshes.new(mesh_name)
        me.from_pydata([v[:] for v in verts], [], faces)

        # UV layers (per-vertex data mapped through loop vertex indices).
        flip_v = options["flip_v"]
        for uv_data, layer_name in ((vdata["uv"], "UVMap"), (vdata["uv2"], "UVMap2")):
            if not uv_data:
                continue
            layer = me.uv_layers.new(name=layer_name)
            coords = [0.0] * (len(me.loops) * 2)
            for li, loop in enumerate(me.loops):
                u, v = uv_data[loop.vertex_index]
                coords[li * 2] = u
                coords[li * 2 + 1] = (1.0 - v) if flip_v else v
            layer.data.foreach_set("uv", coords)

        # Materials + per-face material index from submesh triangle ranges.
        for mat_name in m["materials"]:
            me.materials.append(materials.get(mat_name,
                                              overrides.get(mat_name)))
        if len(m["materials"]) > 1 and m["submeshes"]:
            midx = [0] * len(faces)
            for s in m["submeshes"]:
                mi = min(s["materialIndex"], max(len(m["materials"]) - 1, 0))
                for t in range(s["triangleOffset"],
                               min(s["triangleOffset"] + s["triangleCount"], len(faces))):
                    midx[t] = mi
            me.polygons.foreach_set("material_index", midx)

        me.validate(clean_customdata=False)

        # Smooth shading + custom split normals.
        me.polygons.foreach_set("use_smooth", [True] * len(me.polygons))
        normals = vdata["normals"]
        if normals and options["import_normals"]:
            conv_normals = []
            for nrm in normals:
                v = normal_xform @ Vector(nrm)
                if v.length > 1e-8:
                    v.normalize()
                else:
                    v = Vector((0, 0, 1))
                conv_normals.append(v[:])
            if hasattr(me, "use_auto_smooth"):     # Blender < 4.1
                me.use_auto_smooth = True
                me.auto_smooth_angle = 3.14159
            try:
                me.normals_split_custom_set_from_vertices(conv_normals)
            except RuntimeError:
                pass

        obj_name = node["name"] or mesh_name
        obj = bpy.data.objects.new(obj_name, me)
        context.collection.objects.link(obj)
        obj.parent = arm_obj

        # ------------------------------------------------------------------
        # Skinning
        # ------------------------------------------------------------------
        groups = {}  # bone name -> list of (vert index, weight)

        def add_w(bone, vi, w):
            if w > 0.0:
                groups.setdefault(bone, []).append((vi, w))

        if skinned and vdata["links"] and vdata["weights"]:
            linked = m["linkedNodes"]

            def bone_for_out_index(oi):
                if 0 <= oi < len(linked) and linked[oi] in by_id:
                    return bone_names[by_id[linked[oi]]]
                return None

            default_palette = list(range(len(linked)))
            # per-vertex palette from submesh ranges
            palette_per_vert = [default_palette] * len(positions)
            for s in m["submeshes"]:
                pal = s["indices"] if s["indices"] else default_palette
                for vi in range(s["vertexOffset"],
                                min(s["vertexOffset"] + s["vertexCount"], len(positions))):
                    palette_per_vert[vi] = pal

            links = vdata["links"]
            weights = vdata["weights"]
            for vi in range(len(positions)):
                pal = palette_per_vert[vi]
                for slot in range(4):
                    w = weights[vi][slot]
                    if w <= 0.0:
                        continue
                    li = links[vi][slot]
                    oi = pal[li] if li < len(pal) else li
                    bone = bone_for_out_index(oi)
                    if bone:
                        add_w(bone, vi, w)
        else:
            # rigid bind to this node's own bone
            bone = bone_names[node_index]
            groups[bone] = [(vi, 1.0) for vi in range(len(positions))]

        for bone, entries in groups.items():
            vg = obj.vertex_groups.new(name=bone)
            by_weight = {}
            for vi, w in entries:
                by_weight.setdefault(round(w, 5), []).append(vi)
            for w, idxs in by_weight.items():
                vg.add(idxs, w, "REPLACE")

        mod = obj.modifiers.new(name="Armature", type="ARMATURE")
        mod.object = arm_obj
        return obj

    def import_combinexmesh(context, filepath, options):
        parsed = parse_combinexmesh(filepath)
        nodes = parsed["nodes"]
        overrides = parse_xml_materials(parsed["xml"])

        base = os.path.basename(filepath)
        base = os.path.splitext(base)[0] or base

        # Put everything from this file into its own collection.
        coll = bpy.data.collections.new(base)
        context.scene.collection.children.link(coll)
        layer_coll = None
        for lc in context.view_layer.layer_collection.children:
            if lc.collection == coll:
                layer_coll = lc
                break
        prev_active = context.view_layer.active_layer_collection
        if layer_coll:
            context.view_layer.active_layer_collection = layer_coll
        try:
            if bpy.ops.object.mode_set.poll():
                bpy.ops.object.mode_set(mode="OBJECT")
            bpy.ops.object.select_all(action="DESELECT")

            worlds, by_id = _compute_world_matrices(nodes)
            bone_names = _unique_bone_names(nodes)

            # scale bone display size off the model bounds
            span = 1.0
            pts = [(CONV @ w.translation) for w in
                   (worlds if len(worlds) < 500 else worlds[:500])]
            if pts:
                xs = [p.length for p in pts]
                span = max(max(xs), 1e-3)
            bone_size = max(0.02, min(0.25, span * 0.03)) * options["scale"]

            # apply global scale to bone positions via scaled worlds
            if options["scale"] != 1.0:
                s = Matrix.Scale(options["scale"], 4)
                worlds_scaled = [s @ w for w in worlds]
            else:
                worlds_scaled = worlds

            arm_obj = _build_armature(context, base, nodes, worlds_scaled,
                                      bone_names, bone_size)

            materials = options.get("material_builder")
            if materials is None:
                materials = MaterialBuilder(options["tex_dir"],
                                            options.get("use_shader", True))
            imported = 0
            for i, n in enumerate(nodes):
                m = n["mesh"]
                if not m:
                    continue
                is_collision = (m["vdata"]["uv"] is None
                                and m["vdata"]["normals"] is None)
                if is_collision and not options["import_collision"]:
                    continue
                if _build_mesh_object(context, n, i, nodes, worlds_scaled, by_id,
                                      bone_names, arm_obj, materials, overrides,
                                      options):
                    imported += 1

            # keep the embedded XML for reference
            txt = bpy.data.texts.new(base + ".xml")
            txt.write(parsed["xml"])
            return imported, len(nodes)
        finally:
            if layer_coll:
                context.view_layer.active_layer_collection = prev_active

    _ADDON_KEY = (__package__ or __name__).split(".")[0]

    class CombineXMeshPreferences(bpy.types.AddonPreferences):
        bl_idname = _ADDON_KEY

        tex_dir: bpy.props.StringProperty(
            name="Textures folder",
            description="Base Textures Path should be your editor folder (Ex: F:\\archives\\snowrunner\\editor\\)",
            default="", subtype="DIR_PATH")

        def draw(self, context):
            self.layout.prop(self, "tex_dir")

    def _pref_tex_dir():
        try:
            prefs = bpy.context.preferences.addons[_ADDON_KEY].preferences
            return bpy.path.abspath(prefs.tex_dir) if prefs.tex_dir else ""
        except (KeyError, AttributeError):
            return ""

    class IMPORT_OT_combinexmesh(bpy.types.Operator, ImportHelper):
        """Import SnowRunner/MudRunner CombineXMesh model(s) with rig"""
        bl_idname = "import_scene.combinexmesh"
        bl_label = "Import Import Snowrunner Model ([meshes])"
        bl_options = {"REGISTER", "UNDO"}

        filename_ext = ""
        filter_glob: bpy.props.StringProperty(default="*", options={"HIDDEN"})

        files: bpy.props.CollectionProperty(
            type=bpy.types.OperatorFileListElement, options={"HIDDEN", "SKIP_SAVE"})
        directory: bpy.props.StringProperty(subtype="DIR_PATH",
                                            options={"HIDDEN", "SKIP_SAVE"})

        import_collision: bpy.props.BoolProperty(
            name="Import collision meshes",
            description="Also import position-only collision meshes (cdt)",
            default=False)

        def draw(self, context):
            self.layout.prop(self, "import_collision")

        def execute(self, context):
            options = {
                "import_collision": self.import_collision,
                "import_normals": True,
                "flip_v": True,
                "scale": 1.0,
                "tex_dir": _pref_tex_dir(),
                "use_shader": True,
            }
            # one shared builder: index textures once, reuse materials
            options["material_builder"] = MaterialBuilder(
                options["tex_dir"], True)
            paths = []
            if self.files and self.directory:
                paths = [os.path.join(self.directory, f.name)
                         for f in self.files if f.name]
            if not paths and self.filepath:
                paths = [self.filepath]

            ok, failed = 0, []
            for p in paths:
                try:
                    meshes, node_count = import_combinexmesh(context, p, options)
                    ok += 1
                    print("CombineXMesh: imported %s (%d meshes, %d nodes)"
                          % (os.path.basename(p), meshes, node_count))
                except MeshFormatError as e:
                    failed.append("%s: %s" % (os.path.basename(p), e))
                except Exception as e:  # keep going on multi-import
                    failed.append("%s: %r" % (os.path.basename(p), e))

            if failed:
                self.report({"WARNING" if ok else "ERROR"},
                            "Failed: " + "; ".join(failed))
                return {"FINISHED"} if ok else {"CANCELLED"}
            self.report({"INFO"}, "Imported %d file(s)" % ok)
            return {"FINISHED"}

    class IMPORT_OT_snowrunner_materials(bpy.types.Operator, ImportHelper):
        """Import/refresh materials from SnowRunner material XML file(s).

        Parses <Material .../> (and <MaterialOverride .../>) entries and
        builds or updates Blender materials of the same name using the
        bundled SnowRunner shader, so it can be run after importing meshes
        to upgrade their placeholder materials with full texture sets."""
        bl_idname = "import_scene.snowrunner_materials"
        bl_label = "Import SnowRunner Materials"
        bl_options = {"REGISTER", "UNDO"}

        filename_ext = ".xml"
        filter_glob: bpy.props.StringProperty(default="*.xml;*",
                                              options={"HIDDEN"})
        files: bpy.props.CollectionProperty(
            type=bpy.types.OperatorFileListElement,
            options={"HIDDEN", "SKIP_SAVE"})
        directory: bpy.props.StringProperty(subtype="DIR_PATH",
                                            options={"HIDDEN", "SKIP_SAVE"})

        only_existing: bpy.props.BoolProperty(
            name="Only update existing materials",
            description="Skip materials that aren't already used in this "
                        "blend file",
            default=False)

        def draw(self, context):
            self.layout.prop(self, "only_existing")

        def execute(self, context):
            paths = []
            if self.files and self.directory:
                paths = [os.path.join(self.directory, f.name)
                         for f in self.files if f.name]
            if not paths and self.filepath:
                paths = [self.filepath]

            builder = MaterialBuilder(_pref_tex_dir(), True)
            count = 0
            for p in paths:
                try:
                    with open(p, "r", encoding="cp1251", errors="replace") as fh:
                        xml = fh.read()
                except OSError as e:
                    self.report({"WARNING"}, "%s: %s" % (os.path.basename(p), e))
                    continue
                for name, props in parse_xml_materials(xml).items():
                    if self.only_existing and name not in bpy.data.materials:
                        continue
                    builder.get(name, props)
                    count += 1
            if count == 0:
                self.report({"WARNING"}, "No materials found in selected file(s)")
                return {"CANCELLED"}
            self.report({"INFO"}, "Built/updated %d material(s)" % count)
            return {"FINISHED"}

    def menu_func_import(self, context):
        self.layout.operator(IMPORT_OT_combinexmesh.bl_idname,
                             text="Import Snowrunner Model ([meshes])")

    classes = (CombineXMeshPreferences, IMPORT_OT_combinexmesh,
               IMPORT_OT_snowrunner_materials)

    def register():
        for c in classes:
            bpy.utils.register_class(c)
        bpy.types.TOPBAR_MT_file_import.append(menu_func_import)

    def unregister():
        bpy.types.TOPBAR_MT_file_import.remove(menu_func_import)
        for c in classes:
            bpy.utils.unregister_class(c)


if __name__ == "__main__" and HAS_BPY:
    register()
