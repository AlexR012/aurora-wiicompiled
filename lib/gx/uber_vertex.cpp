#include "ubershader.hpp"

namespace aurora::gx {

std::string uber_vertex_wgsl() {
  std::string source;
  const auto define = [&source](std::string_view name, u32 value) {
    source += fmt::format("const v_{} = {}u;\n", name, value);
  };
#define V_ENUM(name) define(#name, static_cast<u32>(name))
  V_ENUM(GX_NONE);
  V_ENUM(GX_DIRECT);
  V_ENUM(GX_INDEX8);
  V_ENUM(GX_INDEX16);
  V_ENUM(GX_VA_PNMTXIDX);
  V_ENUM(GX_VA_TEX0MTXIDX);
  V_ENUM(GX_VA_TEX7MTXIDX);
  V_ENUM(GX_VA_POS);
  V_ENUM(GX_VA_NRM);
  V_ENUM(GX_VA_CLR0);
  V_ENUM(GX_VA_CLR1);
  V_ENUM(GX_VA_TEX0);
  V_ENUM(GX_U8);
  V_ENUM(GX_S8);
  V_ENUM(GX_U16);
  V_ENUM(GX_S16);
  V_ENUM(GX_F32);
  V_ENUM(GX_RGB565);
  V_ENUM(GX_RGB8);
  V_ENUM(GX_RGBX8);
  V_ENUM(GX_RGBA4);
  V_ENUM(GX_RGBA6);
  V_ENUM(GX_RGBA8);
  V_ENUM(GX_SRC_VTX);
  V_ENUM(GX_AF_NONE);
  V_ENUM(GX_AF_SPOT);
  V_ENUM(GX_AF_SPEC);
  V_ENUM(GX_DF_NONE);
  V_ENUM(GX_DF_SIGN);
  V_ENUM(GX_DF_CLAMP);
  V_ENUM(GX_ALPHA0);
  V_ENUM(GX_TG_MTX2x4);
  V_ENUM(GX_TG_MTX3x4);
  V_ENUM(GX_TG_SRTG);
  V_ENUM(GX_TG_BUMP0);
  V_ENUM(GX_TG_BUMP7);
  V_ENUM(GX_TG_POS);
  V_ENUM(GX_TG_NRM);
  V_ENUM(GX_TG_BINRM);
  V_ENUM(GX_TG_TANGENT);
  V_ENUM(GX_TG_COLOR0);
  V_ENUM(GX_TG_COLOR1);
  V_ENUM(GX_TG_TEX0);
  V_ENUM(GX_TG_TEX7);
  V_ENUM(GX_TG_TEXCOORD0);
  V_ENUM(GX_TG_TEXCOORD6);
  V_ENUM(GX_MAX_TEXGENSRC);
  V_ENUM(GX_IDENTITY);
  V_ENUM(GX_PTTEXMTX0);
  V_ENUM(GX_PTIDENTITY);
#undef V_ENUM
  define("MaxLights", GX::MaxLights);
  define("MaxPostexMtx", MaxPnMtx + MaxTexMtx);
  define("MaxPTTexMtx", MaxPTTexMtx);
  source += R"""(
fn v_attr_config(attr: u32) -> u32 {
  return ShaderConfig_attrs + attr * AttrConfig_size;
}

fn v_attr_type(attr: u32) -> u32 {
  return cfg_u8(v_attr_config(attr) + AttrConfig_attrType);
}

fn v_component_size(comp: u32) -> u32 {
  switch comp {
    case v_GX_U8, v_GX_S8: { return 1u; }
    case v_GX_U16, v_GX_S16: { return 2u; }
    default: { return 4u; }
  }
}

fn v_fetch_scalar(p: ptr<storage, array<u32>>, offset: u32, comp: u32, frac: u32, le: bool) -> f32 {
  switch comp {
    case v_GX_U8: { return fetch_u8_1(p, offset, frac, le); }
    case v_GX_S8: { return fetch_s8_1(p, offset, frac, le); }
    case v_GX_U16: { return fetch_u16_1(p, offset, frac, le); }
    case v_GX_S16: { return fetch_s16_1(p, offset, frac, le); }
    default: { return fetch_f32_1(p, offset, le); }
  }
}

fn v_fetch_attr(p: ptr<storage, array<u32>>, offset: u32, attr: u32,
                comp: u32, count: u32, frac: u32, le: bool) -> vec4f {
  if (attr == v_GX_VA_PNMTXIDX) {
    return vec4f(f32((raw_fetch_u8_1(p, offset) & 0x3fu) / 3u), 0.0, 0.0, 0.0);
  }
  if (attr >= v_GX_VA_TEX0MTXIDX && attr <= v_GX_VA_TEX7MTXIDX) {
    return vec4f(f32(raw_fetch_u8_1(p, offset) & 0x3fu), 0.0, 0.0, 0.0);
  }
  if (attr == v_GX_VA_CLR0 || attr == v_GX_VA_CLR1) {
    switch comp {
      case v_GX_RGB565: { return fetch_rgb565(p, offset, le); }
      case v_GX_RGB8: { return fetch_rgb8(p, offset, le); }
      case v_GX_RGBX8: { return fetch_rgbx8(p, offset, le); }
      case v_GX_RGBA4: { return fetch_rgba4(p, offset, le); }
      case v_GX_RGBA6: { return fetch_rgba6(p, offset, le); }
      default: { return fetch_rgba8(p, offset, le); }
    }
  }
  if (comp == v_GX_RGBA8) {
    return unpack4x8unorm(load_u32_raw(p, offset));
  }
  var result = vec4f(0.0);
  let component_size = v_component_size(comp);
  for (var i = 0u; i < min(count, 4u); i++) {
    result[i] = v_fetch_scalar(p, offset + i * component_size, comp, frac, le);
  }
  return result;
}

// FIFO indices are big endian; array payloads use AttrConfig.le.
fn v_attr(attr: u32, vertex: u32, normal_group: u32) -> vec4f {
  let mapping = v_attr_config(attr);
  let kind = cfg_u8(mapping + AttrConfig_attrType);
  if (kind == v_GX_NONE) {
    if (attr == v_GX_VA_PNMTXIDX) {
      return vec4f(f32(imm.current_pnmtx), 0.0, 0.0, 0.0);
    }
    if (attr == v_GX_VA_NRM) {
      return vec4f(1.0, 0.0, 0.0, 0.0);
    }
    if (attr == v_GX_VA_CLR0 || attr == v_GX_VA_CLR1) {
      return vec4f(1.0);
    }
    return vec4f(0.0);
  }
  let comp = cfg_u8(mapping + AttrConfig_compType);
  let frac = cfg_u8(mapping + AttrConfig_frac);
  var count = cfg_u8(mapping + AttrConfig_cnt);
  var offset = imm.vtx_start + vertex * cfg_u8(ShaderConfig_vtxStride) + cfg_u8(mapping + AttrConfig_offset);
  var group_offset = 0u;
  if (attr == v_GX_VA_NRM && count == 9u) {
    count = 3u;
    if (cfg_u8(mapping + AttrConfig_nbt3) != 0u &&
        (kind == v_GX_INDEX8 || kind == v_GX_INDEX16)) {
      offset += normal_group * select(1u, 2u, kind == v_GX_INDEX16);
    }
    group_offset = normal_group * 3u * v_component_size(comp);
  }
  if (kind == v_GX_INDEX8 || kind == v_GX_INDEX16) {
    var index: u32;
    if (kind == v_GX_INDEX8) {
      index = raw_fetch_u8_1(&vbuf, offset);
    } else {
      index = raw_fetch_u16_1(&vbuf, offset, false);
    }
    let array_offset = array_start(attr - v_GX_VA_POS);
    offset = array_offset + index * cfg_u8(mapping + AttrConfig_stride) + group_offset;
    return v_fetch_attr(&abuf, offset, attr, comp, count, frac, cfg_u8(mapping + AttrConfig_le) != 0u);
  }
  return v_fetch_attr(&vbuf, offset + group_offset, attr, comp, count, frac, false);
}

fn v_postex_matrix(index: u32) -> mat3x4f {
  return uniform_mat3x4(uniform_offset(UberUniformOffsets_postex) + index * 48u);
}

fn v_normal_matrix(index: u32) -> mat3x4f {
  return uniform_mat3x4(uniform_offset(UberUniformOffsets_nrm) + index * 48u);
}

fn v_vertex_color(channel: u32, vertex: u32) -> vec4f {
  let has_color0 = v_attr_type(v_GX_VA_CLR0) != v_GX_NONE;
  let has_color1 = v_attr_type(v_GX_VA_CLR1) != v_GX_NONE;
  if (channel == 0u) {
    if (has_color0) { return v_attr(v_GX_VA_CLR0, vertex, 0u); }
    if (has_color1) { return v_attr(v_GX_VA_CLR1, vertex, 0u); }
  } else if (has_color0 && has_color1) {
    return v_attr(v_GX_VA_CLR1, vertex, 0u);
  }
  return vec4f(1.0);
}

fn v_light_factor(channel: u32, light: u32, mv_pos: vec3f, mv_nrm: vec3f) -> f32 {
  let attn_fn = cfg_u32(channel + ColorChannelConfig_attnFn);
  let diff_fn = cfg_u32(channel + ColorChannelConfig_diffFn);
  var ldir = uniform_vec4(light).xyz - mv_pos;
  let dist2 = dot(ldir, ldir);
  let dist = sqrt(dist2);
  var attn = 1.0;
  if (attn_fn == v_GX_AF_NONE) {
    ldir = select(mv_nrm, ldir / max(dist, 1e-20), dist > 0.0);
  } else {
    // Preserve the specialized attenuation behavior, including zero-distance divisions.
    ldir = ldir / dist;
    let direction = uniform_vec4(light + 16u).xyz;
    let cos_att = uniform_vec4(light + 48u).xyz;
    var dist_att = uniform_vec4(light + 64u).xyz;
    if (attn_fn == v_GX_AF_SPOT) {
      let cosine = max(0.0, dot(ldir, direction));
      let cos_attn = dot(cos_att, vec3f(1.0, cosine, cosine * cosine));
      let dist_attn = dot(dist_att, vec3f(1.0, dist, dist2));
      attn = max(0.0, cos_attn / dist_attn);
    } else if (attn_fn == v_GX_AF_SPEC) {
      attn = select(0.0, max(0.0, dot(mv_nrm, direction)), dot(mv_nrm, ldir) >= 0.0);
      let cos_attn = dot(cos_att, vec3f(1.0, attn, attn * attn));
      if (diff_fn != v_GX_DF_NONE) { dist_att = normalize(dist_att); }
      let dist_attn = dot(dist_att, vec3f(1.0, attn, attn * attn));
      attn = select(select(0.0, 1.0, cos_attn > 0.0), max(0.0, cos_attn / dist_attn), dist_attn != 0.0);
    }
  }
  var diff = 1.0;
  if (diff_fn == v_GX_DF_SIGN) {
    diff = dot(ldir, mv_nrm);
  } else if (diff_fn == v_GX_DF_CLAMP) {
    diff = max(0.0, dot(ldir, mv_nrm));
  }
  return attn * diff;
}
)""";

  // Fixed RGB/scalar entry points keep alpha lighting independent without computing unused RGB lanes.
  // This does not depend on a draw's shader configuration.
  for (bool alpha : {false, true}) {
    source += fmt::format(R"""(
fn v_lighting_{0}(channel_index: u32, vertex_color: vec4f, mv_pos: vec3f, mv_nrm: vec3f) -> {1} {{
  let channel = ShaderConfig_colorChannels + channel_index * ColorChannelConfig_size;
  var material_color = vertex_color;
  if (cfg_u32(channel + ColorChannelConfig_matSrc) != v_GX_SRC_VTX) {{
    material_color = uniform_vec4(uniform_offset(UberUniformOffsets_material + channel_index * 4u));
  }}
  let material = {2}(round(material_color.{3} * 255.0));
  if (cfg_u8(channel + ColorChannelConfig_lightingEnabled) == 0u) {{
    return {1}(material) / 255.0;
  }}
  var ambient_color = vertex_color;
  if (cfg_u32(channel + ColorChannelConfig_ambSrc) != v_GX_SRC_VTX) {{
    ambient_color = uniform_vec4(uniform_offset(UberUniformOffsets_ambient + channel_index * 4u));
  }}
  var lighting = {2}(round(ambient_color.{3} * 255.0));
  let light_mask = uniform_u32(uniform_offset(UberUniformOffsets_lightState) + channel_index * 4u);
  let lights = uniform_offset(UberUniformOffsets_lights);
  for (var i = 0u; i < v_MaxLights; i++) {{
    if ((light_mask & (1u << i)) == 0u) {{ continue; }}
    let light = lights + i * 80u;
    let factor = v_light_factor(channel, light, mv_pos, mv_nrm);
    lighting += {2}(round(factor * uniform_vec4(light + 32u).{3} * 255.0));
  }}
  let lacc = clamp(lighting, {2}(0), {2}(255));
  return {1}((material * (lacc + (lacc >> {4}))) >> {5}) / 255.0;
}}
)""",
                          alpha ? "alpha" : "rgb", alpha ? "f32" : "vec3f", alpha ? "i32" : "vec3i",
                          alpha ? "a" : "rgb", alpha ? "7u" : "vec3u(7u)", alpha ? "8u" : "vec3u(8u)");
  }

  source += R"""(
@vertex
fn vs_main(@builtin(vertex_index) vidx: u32, @builtin(instance_index) iidx: u32) -> UberVertexOutput {
  var out: UberVertexOutput;
  let flags = cfg_u32(UberShaderState_flags);
  let line_mode = flags & 3u;
  let proj = uniform_mat4x4(uniform_offset(UberUniformOffsets_projection));
  let render_viewport = vec2f(uniform_f32(0u), uniform_f32(4u));
  let logical_viewport = vec2f(uniform_f32(8u), uniform_f32(12u));
  var vertex = vidx;
  if (line_mode == 3u) {
    vertex = iidx;
  } else if (line_mode != 0u) {
    vertex = iidx * select(1u, 2u, line_mode == 1u) + select(0u, 1u, vidx >= 2u);
  }
  let in_pos = v_attr(v_GX_VA_POS, vertex, 0u).xyz;
  let pnmtx = u32(v_attr(v_GX_VA_PNMTXIDX, vertex, 0u).x);
  let mv_pos = vec4f(in_pos, 1.0) * v_postex_matrix(pnmtx);
  let clip = vec4f(mv_pos, 1.0) * proj;
  out.pos = clip;
  if (line_mode != 0u) {
    let viewport_scale = render_viewport / max(logical_viewport, vec2f(1.0));
    let width = uniform_f32(16u) * min(viewport_scale.x, viewport_scale.y);
    var offset_px: vec2f;
    if (line_mode == 3u) {
      let x_sign = select(-1.0, 1.0, (vidx & 1u) != 0u);
      let y_sign = select(-1.0, 1.0, vidx >= 2u);
      offset_px = vec2f(x_sign, y_sign) * (width / 2.0);
    } else {
      let other_vertex = select(vertex + 1u, vertex - 1u, vidx >= 2u);
      let other_pos = v_attr(v_GX_VA_POS, other_vertex, 0u).xyz;
      let other_pnmtx = u32(v_attr(v_GX_VA_PNMTXIDX, other_vertex, 0u).x);
      let other_mv_pos = vec4f(other_pos, 1.0) * v_postex_matrix(other_pnmtx);
      let other_clip = vec4f(other_mv_pos, 1.0) * proj;
      let ndc_a = select(clip.xy / clip.w, other_clip.xy / other_clip.w, vidx >= 2u);
      let ndc_b = select(other_clip.xy / other_clip.w, clip.xy / clip.w, vidx >= 2u);
      let delta_px = (ndc_b - ndc_a) / 2.0 * render_viewport;
      let dir_px = select(vec2f(1.0, 0.0), normalize(delta_px), dot(delta_px, delta_px) > 1e-10);
      let perp_px = vec2f(-dir_px.y, dir_px.x);
      offset_px = perp_px * (width / 2.0) * select(-1.0, 1.0, (vidx & 1u) != 0u);
    }
    let offset_ndc = (offset_px * 2.0) / render_viewport;
    out.pos = vec4f(clip.xy + offset_ndc * clip.w, clip.zw);
  }
  let in_nrm = v_attr(v_GX_VA_NRM, vertex, 0u).xyz;
  let normal_matrix = v_normal_matrix(pnmtx);
  let nrm_tmp = vec4f(in_nrm, 0.0) * normal_matrix;
  let mv_nrm = select(nrm_tmp, normalize(nrm_tmp), dot(nrm_tmp, nrm_tmp) > 1e-10);
  out.mv_nrm = mv_nrm;
  let color_mask = cfg_u32(UberShaderState_sampledColorChannels);
  if ((color_mask & 1u) != 0u) {
    let color = v_vertex_color(0u, vertex);
    out.cc0 = vec4f(v_lighting_rgb(0u, color, mv_pos, mv_nrm),
                   v_lighting_alpha(v_GX_ALPHA0, color, mv_pos, mv_nrm));
  }
  if ((color_mask & 2u) != 0u) {
    let color = v_vertex_color(1u, vertex);
    out.cc1 = vec4f(v_lighting_rgb(1u, color, mv_pos, mv_nrm),
                   v_lighting_alpha(v_GX_ALPHA0 + 1u, color, mv_pos, mv_nrm));
  }

  var coords: array<vec3f, 8>;
  var inputs: array<vec4f, 8>;
  let texcoord_mask = cfg_u32(UberShaderState_sampledTexCoords);
  for (var i = 0u; i < 8u; i++) {
    if ((texcoord_mask & (1u << i)) == 0u) { continue; }
    let tcg = ShaderConfig_tcgs + i * TcgConfig_size;
    let kind = cfg_u32(tcg + TcgConfig_type);
    let src = cfg_u32(tcg + TcgConfig_src);
    let is_matrix = kind == v_GX_TG_MTX2x4 || kind == v_GX_TG_MTX3x4;
    var transformed: vec3f;
    if (kind >= v_GX_TG_BUMP0 && kind <= v_GX_TG_BUMP7) {
      // Emboss consumes a prior texgen, and scales (rather than normalizes) the NBT basis.
      let light = uniform_offset(UberUniformOffsets_lights) + (kind - v_GX_TG_BUMP0) * 80u;
      let light_dir = normalize(uniform_vec4(light).xyz - mv_pos);
      let tangent = vec4f(v_attr(v_GX_VA_NRM, vertex, 2u).xyz, 0.0) * normal_matrix;
      let binormal = vec4f(v_attr(v_GX_VA_NRM, vertex, 1u).xyz, 0.0) * normal_matrix;
      transformed = coords[cfg_u32(tcg + TcgConfig_embossSrc)] +
                    vec3f(dot(light_dir, tangent), dot(light_dir, binormal), 0.0);
      coords[i] = vec3f(transformed.xy, 1.0);
      continue;
    } else {
      var coord = vec4f(0.0, 0.0, 1.0, 1.0);
      if (src >= v_GX_TG_TEX0 && src <= v_GX_TG_TEX7) {
        coord = vec4f(v_attr(v_GX_VA_TEX0 + src - v_GX_TG_TEX0, vertex, 0u).xy, 1.0, 1.0);
      } else if (src == v_GX_TG_POS) {
        coord = vec4f(in_pos, 1.0);
      } else if (src == v_GX_TG_NRM) {
        coord = vec4f(in_nrm, 1.0);
      } else if (src == v_GX_TG_BINRM) {
        coord = vec4f(v_attr(v_GX_VA_NRM, vertex, 1u).xyz, 1.0);
      } else if (src == v_GX_TG_TANGENT) {
        coord = vec4f(v_attr(v_GX_VA_NRM, vertex, 2u).xyz, 1.0);
      } else if (src == v_GX_TG_COLOR0) {
        coord = out.cc0;
      } else if (src == v_GX_TG_COLOR1) {
        coord = out.cc1;
      } else if (src >= v_GX_TG_TEXCOORD0 && src <= v_GX_TG_TEXCOORD6) {
        coord = inputs[src - v_GX_TG_TEXCOORD0];
      }
      if (cfg_u8(tcg + TcgConfig_inputFormAB11) != 0u) { coord.z = 1.0; }
      inputs[i] = coord;
      transformed = coord.xyz;
      if (is_matrix) {
        let matrix_attr = v_GX_VA_TEX0MTXIDX + i;
        let matrix_attr_type = v_attr_type(matrix_attr);
        if (matrix_attr_type == v_GX_DIRECT || matrix_attr_type == v_GX_INDEX8 || matrix_attr_type == v_GX_INDEX16) {
          let matrix = u32(v_attr(matrix_attr, vertex, 0u).x) / 3u;
          transformed = coord * v_postex_matrix(matrix);
        } else {
          let matrix = cfg_u32(tcg + TcgConfig_mtx);
          if (matrix != v_GX_IDENTITY) {
            transformed = coord * v_postex_matrix(matrix / 3u);
          }
        }
        if (kind == v_GX_TG_MTX2x4) { transformed.z = 1.0; }
      } else if (kind == v_GX_TG_SRTG) {
        transformed = vec3f(coord.xy, 1.0);
      }
    }
    {
      if (cfg_u8(tcg + TcgConfig_normalize) != 0u) { transformed = normalize(transformed); }
      let post_matrix = cfg_u32(tcg + TcgConfig_postMtx);
      if (post_matrix != v_GX_PTIDENTITY) {
        let index = (post_matrix - v_GX_PTTEXMTX0) / 3u;
        transformed = vec4f(transformed, 1.0) *
                      uniform_mat3x4(uniform_offset(UberUniformOffsets_postmtx) + index * 48u);
      }
    }
    if (kind == v_GX_TG_MTX3x4 && transformed.z == 0.0) {
      transformed.x = clamp(transformed.x / 2.0, -1.0, 1.0);
      transformed.y = clamp(transformed.y / 2.0, -1.0, 1.0);
    }
    if (line_mode != 0u && (uniform_u32(28u) & (1u << i)) != 0u) {
      let tex_offset = uniform_f32(24u);
      if (line_mode == 3u) {
        if ((vidx & 1u) != 0u) { transformed.x += tex_offset; }
        if (vidx >= 2u) { transformed.y += tex_offset; }
      } else if ((vidx & 1u) != 0u) { transformed.y += tex_offset; }
    }
    coords[i] = transformed;
  }
  out.tex0 = coords[0];
  out.tex1 = coords[1];
  out.tex2 = coords[2];
  out.tex3 = coords[3];
  out.tex4 = coords[4];
  out.tex5 = coords[5];
  out.tex6 = coords[6];
  out.tex7 = coords[7];
  return out;
}
)""";
  return source;
}

} // namespace aurora::gx
