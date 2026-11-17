#include "ubershader.hpp"

#include <dolphin/gx/GXEnum.h>

namespace aurora::gx {

std::string uber_fragment_wgsl(const UberModuleKey& key) {
  std::string source;
#define F_ENUM(value) source += "const f_" #value ": u32 = " + std::to_string(static_cast<u32>(value)) + "u;\n"
  F_ENUM(GX_MAX_TEXMAP);
  F_ENUM(GX_MAX_TEXCOORD);
  F_ENUM(GX_MAX_INDTEXSTAGE);
  F_ENUM(GX_TG_MTX3x4);
  F_ENUM(GX_CC_A2);
  F_ENUM(GX_CC_TEXC);
  F_ENUM(GX_CC_TEXA);
  F_ENUM(GX_CC_RASC);
  F_ENUM(GX_CC_RASA);
  F_ENUM(GX_CC_ONE);
  F_ENUM(GX_CC_HALF);
  F_ENUM(GX_CC_KONST);
  F_ENUM(GX_CA_A2);
  F_ENUM(GX_CA_TEXA);
  F_ENUM(GX_CA_RASA);
  F_ENUM(GX_CA_KONST);
  F_ENUM(GX_TEV_KCSEL_1_8);
  F_ENUM(GX_TEV_KCSEL_K0);
  F_ENUM(GX_TEV_KCSEL_K3);
  F_ENUM(GX_TEV_KCSEL_K0_R);
  F_ENUM(GX_TEV_KCSEL_K3_A);
  F_ENUM(GX_TEV_KASEL_1_8);
  F_ENUM(GX_TEV_KASEL_K0_R);
  F_ENUM(GX_TEV_KASEL_K3_A);
  F_ENUM(GX_COLOR0);
  F_ENUM(GX_COLOR1);
  F_ENUM(GX_ALPHA0);
  F_ENUM(GX_ALPHA1);
  F_ENUM(GX_COLOR0A0);
  F_ENUM(GX_COLOR1A1);
  F_ENUM(GX_ALPHA_BUMP);
  F_ENUM(GX_ALPHA_BUMPN);
  F_ENUM(GX_TEV_ADD);
  F_ENUM(GX_TEV_SUB);
  F_ENUM(GX_TEV_COMP_R8_GT);
  F_ENUM(GX_TEV_COMP_R8_EQ);
  F_ENUM(GX_TEV_COMP_GR16_GT);
  F_ENUM(GX_TEV_COMP_GR16_EQ);
  F_ENUM(GX_TEV_COMP_BGR24_GT);
  F_ENUM(GX_TEV_COMP_BGR24_EQ);
  F_ENUM(GX_TEV_COMP_RGB8_GT);
  F_ENUM(GX_TEV_COMP_RGB8_EQ);
  F_ENUM(GX_CS_SCALE_2);
  F_ENUM(GX_CS_SCALE_4);
  F_ENUM(GX_CS_DIVIDE_2);
  F_ENUM(GX_TB_ADDHALF);
  F_ENUM(GX_TB_SUBHALF);
  F_ENUM(GX_ITF_8);
  F_ENUM(GX_ITF_5);
  F_ENUM(GX_ITF_4);
  F_ENUM(GX_ITF_3);
  F_ENUM(GX_ITBA_OFF);
  F_ENUM(GX_ITM_OFF);
  F_ENUM(GX_ITM_0);
  F_ENUM(GX_ITM_2);
  F_ENUM(GX_ITM_S0);
  F_ENUM(GX_ITM_S2);
  F_ENUM(GX_ITM_T0);
  F_ENUM(GX_ITM_T2);
  F_ENUM(GX_ITW_OFF);
  F_ENUM(GX_ITW_256);
  F_ENUM(GX_ITW_0);
  F_ENUM(GX_ZT_DISABLE);
  F_ENUM(GX_ZT_ADD);
  F_ENUM(GX_FOG_NONE);
  F_ENUM(GX_FOG_PERSP_EXP);
  F_ENUM(GX_FOG_PERSP_EXP2);
  F_ENUM(GX_FOG_PERSP_REVEXP);
  F_ENUM(GX_FOG_PERSP_REVEXP2);
  F_ENUM(GX_NEVER);
  F_ENUM(GX_LESS);
  F_ENUM(GX_EQUAL);
  F_ENUM(GX_LEQUAL);
  F_ENUM(GX_GREATER);
  F_ENUM(GX_NEQUAL);
  F_ENUM(GX_GEQUAL);
  F_ENUM(GX_ALWAYS);
  F_ENUM(GX_AOP_OR);
  F_ENUM(GX_AOP_XOR);
  F_ENUM(GX_AOP_XNOR);
#undef F_ENUM
  source += UseReversedZ ? "const f_reversed_z = true;\n" : "const f_reversed_z = false;\n";
  source += R"(
fn f_sample(map: u32, uv: vec2f, bias: f32) -> vec4f {
  switch map {
    case 0u: { return textureSampleBias(tex0, tex0_samp, uv, bias); }
    case 1u: { return textureSampleBias(tex1, tex1_samp, uv, bias); }
    case 2u: { return textureSampleBias(tex2, tex2_samp, uv, bias); }
    case 3u: { return textureSampleBias(tex3, tex3_samp, uv, bias); }
    case 4u: { return textureSampleBias(tex4, tex4_samp, uv, bias); }
    case 5u: { return textureSampleBias(tex5, tex5_samp, uv, bias); }
    case 6u: { return textureSampleBias(tex6, tex6_samp, uv, bias); }
    case 7u: { return textureSampleBias(tex7, tex7_samp, uv, bias); }
    default: { return vec4f(0.0); }
  }
}

fn f_pass(offset: u32) -> vec4u {
  return vec4u(cfg_u32(offset + TevPass_a), cfg_u32(offset + TevPass_b),
               cfg_u32(offset + TevPass_c), cfg_u32(offset + TevPass_d));
}

fn f_effective_texcoord(coord: u32, count: u32) -> u32 {
  return select(0u, coord, coord < count && coord < f_GX_MAX_TEXCOORD);
}

fn f_swap(value: vec4f, index: u32) -> vec4f {
  let offset = ShaderConfig_tevSwapTable + index * TevSwap_size;
  return vec4f(value[cfg_u32(offset + TevSwap_red)], value[cfg_u32(offset + TevSwap_green)],
               value[cfg_u32(offset + TevSwap_blue)], value[cfg_u32(offset + TevSwap_alpha)]);
}

fn f_kcolor(selection: u32) -> vec3f {
  if (selection <= f_GX_TEV_KCSEL_1_8) {
    return vec3f(f32(8u - selection) / 8.0);
  }
  if (selection >= f_GX_TEV_KCSEL_K0 && selection <= f_GX_TEV_KCSEL_K3) {
    return uniform_vec4(uniform_offset(UberUniformOffsets_kcolor +
                        (selection - f_GX_TEV_KCSEL_K0) * 4u)).rgb;
  }
  if (selection < f_GX_TEV_KCSEL_K0_R || selection > f_GX_TEV_KCSEL_K3_A) { return vec3f(0.0); }
  let component = selection - f_GX_TEV_KCSEL_K0_R;
  let value = uniform_vec4(uniform_offset(UberUniformOffsets_kcolor + (component & 3u) * 4u));
  return vec3f(value[component >> 2u]);
}

fn f_kalpha(selection: u32) -> f32 {
  if (selection <= f_GX_TEV_KASEL_1_8) {
    return f32(8u - selection) / 8.0;
  }
  if (selection < f_GX_TEV_KASEL_K0_R || selection > f_GX_TEV_KASEL_K3_A) { return 0.0; }
  let component = selection - f_GX_TEV_KASEL_K0_R;
  let value = uniform_vec4(uniform_offset(UberUniformOffsets_kcolor + (component & 3u) * 4u));
  return value[component >> 2u];
}

fn f_color_arg(arg: u32, regs: ptr<function, array<vec4f, 4>>, tex: vec4f,
               ras: vec4f, konst: vec3f) -> vec3f {
  // GX alternates RGB and replicated-alpha register selectors.
  if (arg <= f_GX_CC_A2) {
    let value = (*regs)[arg >> 1u];
    return select(value.rgb, vec3f(value.a), (arg & 1u) != 0u);
  }
  switch arg {
    case f_GX_CC_TEXC: { return tex.rgb; }
    case f_GX_CC_TEXA: { return vec3f(tex.a); }
    case f_GX_CC_RASC: { return ras.rgb; }
    case f_GX_CC_RASA: { return vec3f(ras.a); }
    case f_GX_CC_ONE: { return vec3f(1.0); }
    case f_GX_CC_HALF: { return vec3f(0.5); }
    case f_GX_CC_KONST: { return konst; }
    default: { return vec3f(0.0); }
  }
}

fn f_alpha_arg(arg: u32, regs: ptr<function, array<vec4f, 4>>, tex: vec4f,
               ras: vec4f, konst: f32) -> f32 {
  if (arg <= f_GX_CA_A2) {
    return (*regs)[arg].a;
  }
  switch arg {
    case f_GX_CA_TEXA: { return tex.a; }
    case f_GX_CA_RASA: { return ras.a; }
    case f_GX_CA_KONST: { return konst; }
    default: { return 0.0; }
  }
}

fn f_color_input(arg:u32, regs:ptr<function,array<vec4f,4>>, tex:vec4f, ras:vec4f,
                 konst:vec3f, rgb_mask:u32, alpha_mask:u32)->vec3f {
  let value = f_color_arg(arg, regs, tex, ras, konst);
  if (arg <= f_GX_CC_A2) {
    let mask = select(rgb_mask, alpha_mask, (arg & 1u) != 0u);
    if ((mask & (1u << (arg >> 1u))) == 0u) { return tev_overflow_vec3f(value); }
  }
  return value;
}
fn f_alpha_input(arg:u32, regs:ptr<function,array<vec4f,4>>, tex:vec4f, ras:vec4f,
                 konst:f32, mask:u32)->f32 {
  let value = f_alpha_arg(arg, regs, tex, ras, konst);
  if (arg <= f_GX_CA_A2 && (mask & (1u << arg)) == 0u) { return tev_overflow_f32(value); }
  return value;
}
fn f_tev_scale(scale:u32)->f32 {
  switch scale {
    case f_GX_CS_SCALE_2: { return 2.0; }
    case f_GX_CS_SCALE_4: { return 4.0; }
    case f_GX_CS_DIVIDE_2: { return 0.5; }
    default: { return 1.0; }
  }
}
fn tev_regular_vec3f(a:vec3f,b:vec3f,c:vec3f,d:vec3f,bias:f32,scale:u32,sub:bool)->vec3f {
  var value = mix(a,b,c);
  if (sub) { value = -value; }
  return ((value + d) + bias) * f_tev_scale(scale);
}
fn tev_regular_f32(a:f32,b:f32,c:f32,d:f32,bias:f32,scale:u32,sub:bool)->f32 {
  var value = mix(a,b,c);
  if (sub) { value = -value; }
  return ((value + d) + bias) * f_tev_scale(scale);
}
fn tev_s24_fixed_vec2i(value:vec2i)->vec2i { return (value << vec2u(8u)) >> vec2u(8u); }

fn f_tev_bias(bias: u32) -> f32 {
  switch bias {
    case f_GX_TB_ADDHALF: { return 0.5; }
    case f_GX_TB_SUBHALF: { return -0.5; }
    default: { return 0.0; }
  }
}

fn f_compare_rgb(op: u32, a: vec3f, b: vec3f) -> bool {
  // Preserve the specialized shader's packed comparison rounding: round after
  // the dot product, not each color component before packing.
  var lhs = round(a.r * 255.0);
  var rhs = round(b.r * 255.0);
  switch op {
    case f_GX_TEV_COMP_GR16_GT, f_GX_TEV_COMP_GR16_EQ: {
      lhs = round(dot(a.rg * 255.0, vec2f(1.0, 256.0)));
      rhs = round(dot(b.rg * 255.0, vec2f(1.0, 256.0)));
    }
    case f_GX_TEV_COMP_BGR24_GT, f_GX_TEV_COMP_BGR24_EQ: {
      lhs = round(dot(a * 255.0, vec3f(1.0, 256.0, 65536.0)));
      rhs = round(dot(b * 255.0, vec3f(1.0, 256.0, 65536.0)));
    }
    default: {}
  }
  return select(lhs > rhs, lhs == rhs, (op & 1u) != 0u);
}

fn f_color_op(offset: u32, color_a: vec3f, color_b: vec3f,
              color_c: vec3f, d: vec3f) -> vec3f {
  let a = color_a;
  let b = color_b;
  let c = color_c;
  let op = cfg_u32(offset + TevOp_op);
  var result: vec3f;
  switch op {
    case f_GX_TEV_ADD, f_GX_TEV_SUB: {
      result = tev_regular_vec3f(a, b, c, d, f_tev_bias(cfg_u32(offset + TevOp_bias)),
                                cfg_u32(offset + TevOp_scale), op == f_GX_TEV_SUB);
    }
    case f_GX_TEV_COMP_RGB8_GT: {
      result = select(vec3f(0.0), c, round(a * 255.0) > round(b * 255.0)) + d;
    }
    case f_GX_TEV_COMP_RGB8_EQ: {
      result = select(vec3f(0.0), c, round(a * 255.0) == round(b * 255.0)) + d;
    }
    default: {
      result = select(vec3f(0.0), c, f_compare_rgb(op, a, b)) + d;
    }
  }
  if (cfg_u8(offset + TevOp_clamp) != 0u) {
    return clamp(result, vec3f(0.0), vec3f(1.0));
  }
  return clamp(result, vec3f(-4.0), vec3f(4.0));
}

fn f_alpha_op(offset: u32, alpha_a: f32, alpha_b: f32, alpha_c: f32, d: f32,
              color_a: vec3f, color_b: vec3f) -> f32 {
  let a = alpha_a;
  let b = alpha_b;
  let c = alpha_c;
  let op = cfg_u32(offset + TevOp_op);
  var result: f32;
  switch op {
    case f_GX_TEV_ADD, f_GX_TEV_SUB: {
      result = tev_regular_f32(a, b, c, d, f_tev_bias(cfg_u32(offset + TevOp_bias)),
                              cfg_u32(offset + TevOp_scale), op == f_GX_TEV_SUB);
    }
    // RGB8's numeric aliases are A8 comparisons in the alpha combiner.
    case f_GX_TEV_COMP_RGB8_GT: {
      result = select(0.0, c, round(a * 255.0) > round(b * 255.0)) + d;
    }
    case f_GX_TEV_COMP_RGB8_EQ: {
      result = select(0.0, c, round(a * 255.0) == round(b * 255.0)) + d;
    }
    default: {
      result = select(0.0, c, f_compare_rgb(op, color_a, color_b)) + d;
    }
  }
  if (cfg_u8(offset + TevOp_clamp) != 0u) {
    return clamp(result, 0.0, 1.0);
  }
  return clamp(result, -4.0, 4.0);
}

fn f_raster(stage: u32, cc0: vec4f, cc1: vec4f, ind: vec3i, has_ind: bool) -> vec4f {
  let channel = cfg_u32(stage + TevStage_channelId);
  switch channel {
    case f_GX_COLOR0, f_GX_ALPHA0, f_GX_COLOR0A0: {
      return f_swap(cc0, cfg_u32(stage + TevStage_tevSwapRas));
    }
    case f_GX_COLOR1, f_GX_ALPHA1, f_GX_COLOR1A1: {
      return f_swap(cc1, cfg_u32(stage + TevStage_tevSwapRas));
    }
    case f_GX_ALPHA_BUMP, f_GX_ALPHA_BUMPN: {
      let selection = cfg_u32(stage + TevStage_indTexAlphaSel);
      if (!has_ind || selection == f_GX_ITBA_OFF) {
        return vec4f(0.0);
      }
      var mask = 0xf8u;
      switch cfg_u32(stage + TevStage_indTexFormat) {
        case f_GX_ITF_5: { mask = 0xe0u; }
        case f_GX_ITF_4: { mask = 0xf0u; }
        default: {}
      }
      var alpha = f32(u32(ind[selection - 1u]) & mask) / 255.0;
      if (channel == f_GX_ALPHA_BUMPN) {
        alpha *= 255.0 / 248.0;
      }
      // Bump channels bypass the raster swap table.
      return vec4f(alpha);
    }
    default: { return vec4f(0.0); }
  }
}

fn f_uniform_vec4i(offset: u32) -> vec4i {
  return vec4i(bitcast<i32>(uniform_u32(offset)), bitcast<i32>(uniform_u32(offset + 4u)),
               bitcast<i32>(uniform_u32(offset + 8u)), bitcast<i32>(uniform_u32(offset + 12u)));
}

fn f_indirect_offset(stage: u32, sample: vec3i, base: vec2i, has_base: bool) -> vec2i {
  let format = cfg_u32(stage + TevStage_indTexFormat);
  var shift = 0u;
  switch format {
    case f_GX_ITF_5: { shift = 3u; }
    case f_GX_ITF_4: { shift = 4u; }
    case f_GX_ITF_3: { shift = 5u; }
    default: {}
  }
  var coord = sample >> vec3u(shift);
  let bias = cfg_u32(stage + TevStage_indTexBiasSel);
  let bias_value = select(1, -128, format == f_GX_ITF_8);
  coord += vec3i(select(0, bias_value, (bias & 1u) != 0u),
                  select(0, bias_value, (bias & 2u) != 0u),
                  select(0, bias_value, (bias & 4u) != 0u));
  let matrix = cfg_u32(stage + TevStage_indTexMtxId);
  var matrix_index: u32;
  var offset: vec2i;
  if (matrix >= f_GX_ITM_0 && matrix <= f_GX_ITM_2) {
    matrix_index = matrix - f_GX_ITM_0;
    let address = uniform_offset(UberUniformOffsets_indmtx) + matrix_index * 32u;
    let c0 = f_uniform_vec4i(address);
    let c1 = f_uniform_vec4i(address + 16u);
    offset = vec2i((c0.x * coord.x + c0.z * coord.y + c1.x * coord.z) >> 3u,
                   (c0.y * coord.x + c0.w * coord.y + c1.y * coord.z) >> 3u);
  } else if (matrix >= f_GX_ITM_S0 && matrix <= f_GX_ITM_S2 && has_base) {
    matrix_index = matrix - f_GX_ITM_S0;
    offset = (base * vec2i(coord.x)) >> vec2u(8u);
  } else if (matrix >= f_GX_ITM_T0 && matrix <= f_GX_ITM_T2 && has_base) {
    matrix_index = matrix - f_GX_ITM_T0;
    offset = (base * vec2i(coord.y)) >> vec2u(8u);
  } else {
    return vec2i(0);
  }
  let exponent = bitcast<i32>(uniform_u32(uniform_offset(UberUniformOffsets_indmtx) +
                                         matrix_index * 32u + 24u));
  if (exponent >= 0) {
    return offset >> vec2u(u32(exponent));
  }
  return offset << vec2u(u32(-exponent));
}

fn f_wrap(coord: i32, mode: u32) -> i32 {
  if (mode == f_GX_ITW_OFF) {
    return coord;
  }
  if (mode >= f_GX_ITW_0) {
    return 0;
  }
  return coord & i32((32768u >> (mode - f_GX_ITW_256)) - 1u);
}

fn f_fog(color: vec3f, z: f32, screen_x: f32) -> vec3f {
  let kind = cfg_u8(ShaderConfig_fogType);
  if (kind == f_GX_FOG_NONE) { return color; }
  let offset = uniform_offset(UberUniformOffsets_fog);
  let params = uniform_vec4(offset + 16u);
  var ze: f32;
  if ((kind & 8u) == 0u) { ze = params.x / (params.y - z); }
  else { ze = params.x * z; }
  if ((cfg_u32(UberShaderState_flags) & 4u) != 0u) {
    ze *= bitcast<f32>(abuf[imm.fog_range_base + u32(screen_x)]);
  }
  var factor = clamp(ze - params.z, 0.0, 1.0);
  var fog = factor;
  switch (kind & 7u) {
    case f_GX_FOG_PERSP_EXP: { fog = 1.0 - exp2(-8.0 * factor); }
    case f_GX_FOG_PERSP_EXP2: { fog = 1.0 - exp2(-8.0 * factor * factor); }
    case f_GX_FOG_PERSP_REVEXP: { fog = exp2(-8.0 * (1.0 - factor)); }
    case f_GX_FOG_PERSP_REVEXP2: {
      factor = 1.0 - factor;
      fog = exp2(-8.0 * factor * factor);
    }
    default: {}
  }
  return mix(color, uniform_vec4(offset).rgb, clamp(fog, 0.0, 1.0));
}

fn f_alpha_compare(op: u32, alpha: u32, reference: u32) -> bool {
  switch op {
    case f_GX_NEVER: { return false; }
    case f_GX_LESS: { return alpha < reference; }
    case f_GX_EQUAL: { return alpha == reference; }
    case f_GX_LEQUAL: { return alpha <= reference; }
    case f_GX_GREATER: { return alpha > reference; }
    case f_GX_NEQUAL: { return alpha != reference; }
    case f_GX_GEQUAL: { return alpha >= reference; }
    default: { return true; }
  }
}

fn f_alpha_pass(alpha: f32) -> bool {
  let comp0 = cfg_u32(ShaderConfig_alphaCompare + AlphaCompare_comp0);
  let comp1 = cfg_u32(ShaderConfig_alphaCompare + AlphaCompare_comp1);
  // AlphaCompare::operator bool deliberately bypasses the whole operation here.
  if (comp0 == f_GX_ALWAYS && comp1 == f_GX_ALWAYS) {
    return true;
  }
  let value = u32(round(clamp(alpha, 0.0, 1.0) * 255.0));
  let a = f_alpha_compare(comp0, value, cfg_u32(ShaderConfig_alphaCompare + AlphaCompare_ref0));
  let b = f_alpha_compare(comp1, value, cfg_u32(ShaderConfig_alphaCompare + AlphaCompare_ref1));
  switch cfg_u32(ShaderConfig_alphaCompare + AlphaCompare_op) {
    case f_GX_AOP_OR: { return a || b; }
    case f_GX_AOP_XOR: { return a != b; }
    case f_GX_AOP_XNOR: { return a == b; }
    default: { return a && b; }
  }
}
)";
  source += "struct f_FragmentOutput {\n";
  if (key.dstAlphaMode == DstAlphaMode::DualSource) {
    source += "@location(0) @blend_src(0) color:vec4f,\n@location(0) @blend_src(1) blend:vec4f,\n";
  } else {
    source += "@location(0) color:vec4f,\n";
  }
  if (key.normalAttachment != UINT32_MAX)
    source += fmt::format("@location({}) normal:vec4f,\n", key.normalAttachment);
  if (key.depthOutput) source += "@builtin(frag_depth) depth:f32,\n";
  source += "};\n@fragment fn fs_main(in:UberVertexOutput)->f_FragmentOutput {\n";
  source += R"(
  let stage_count = min(cfg_u32(ShaderConfig_tevStageCount), 16u);
  let texgen_count = 8u;
  let ind_count = cfg_u32(ShaderConfig_numIndStages);
  let flags = cfg_u32(UberShaderState_flags);
  let z_op = (flags >> 3u) & 3u;
  var regs: array<vec4f, 4>;
  var rgb_normalized = 0u;
  var alpha_normalized = 0u;
  for (var i = 0u; i < 4u; i++) {
    regs[i] = uniform_vec4(uniform_offset(UberUniformOffsets_tevreg + i * 4u));
  }

  let varyings = array<vec3f, 8>(in.tex0, in.tex1, in.tex2, in.tex3,
                                in.tex4, in.tex5, in.tex6, in.tex7);
  var fixed_coords: array<vec2i, 8>;
  var raw_coords: array<vec2f, 8>;
  let coord_mask = cfg_u32(UberShaderState_sampledTexCoords);
  for (var i = 0u; i < f_GX_MAX_TEXCOORD; i++) {
    if ((coord_mask & (1u << i)) == 0u) {
      continue;
    }
    let coord = varyings[i];
    var uv = coord.xy;
    if (cfg_u32(ShaderConfig_tcgs + i * TcgConfig_size + TcgConfig_type) == f_GX_TG_MTX3x4 &&
        coord.z != 0.0) {
      uv /= coord.z;
    }
    let scale = uniform_vec4(uniform_offset(UberUniformOffsets_texcoordScale + i * 4u));
    raw_coords[i] = uv;
    fixed_coords[i] = vec2i(uv * scale.xy * 128.0);
  }

  let used_ind = cfg_u32(UberShaderState_usedIndStages);
  let z_stage = select(0xffffffffu, stage_count - 1u, z_op != f_GX_ZT_DISABLE);

  // An indirect sample is shared by every TEV stage that references it. Its
  // ABG byte domain is also the unmodified source for alpha-bump selection.
  var indirect: array<vec3i, 4>;
  for (var i = 0u; i < f_GX_MAX_INDTEXSTAGE; i++) {
    if ((used_ind & (1u << i)) == 0u) {
      continue;
    }
    let stage = ShaderConfig_indStages + i * IndStage_size;
    let map = cfg_u32(stage + IndStage_texMapId);
    let coord = f_effective_texcoord(cfg_u32(stage + IndStage_texCoordId), texgen_count);
    var base = vec2i(0);
    if (texgen_count != 0u) {
      base = fixed_coords[coord];
    }
    let scaled = base >> vec2u(cfg_u32(stage + IndStage_scaleS), cfg_u32(stage + IndStage_scaleT));
    let size_bias = uniform_vec4(uniform_offset(UberUniformOffsets_texSizeBias + map * 4u));
    indirect[i] = vec3i(round(255.0 * f_sample(map, vec2f(scaled) / (size_bias.xy * 128.0), size_bias.z).abg));
  }

  var texcoord = vec2i(0);
  var z_sample = vec4f(0.0);
  var last_color = 0u;
  var last_alpha = 0u;
  for (var i = 0u; i < stage_count; i++) {
    let stage = ShaderConfig_tevStages + i * TevStage_size;
    let color_args = f_pass(stage + TevStage_colorPass);
    let alpha_args = f_pass(stage + TevStage_alphaPass);
    let uses_texture = any(color_args == vec4u(f_GX_CC_TEXC)) ||
                       any(color_args == vec4u(f_GX_CC_TEXA)) ||
                       any(alpha_args == vec4u(f_GX_CA_TEXA));
    let map = cfg_u32(stage + TevStage_texMapId);
    let has_coord = cfg_u32(stage + TevStage_texCoordId) < f_GX_MAX_TEXCOORD;
    let can_sample = has_coord && map < f_GX_MAX_TEXMAP;
    let coord = f_effective_texcoord(cfg_u32(stage + TevStage_texCoordId), texgen_count);
    var base = vec2i(0);
    if (has_coord) {
      base = fixed_coords[coord];
    }
    let matrix = cfg_u32(stage + TevStage_indTexMtxId);
    let wrap_s = cfg_u32(stage + TevStage_indTexWrapS);
    let wrap_t = cfg_u32(stage + TevStage_indTexWrapT);
    let add_prev = cfg_u8(stage + TevStage_indTexAddPrev) != 0u;
    var needs_fixed = matrix != f_GX_ITM_OFF || wrap_s != f_GX_ITW_OFF ||
                      wrap_t != f_GX_ITW_OFF || add_prev;
    if (i + 1u < stage_count) {
      needs_fixed = needs_fixed || cfg_u8(stage + TevStage_size + TevStage_indTexAddPrev) != 0u;
    }
    let ind_index = cfg_u32(stage + TevStage_indTexStage);
    let has_ind = ind_index < ind_count;
    var ind = vec3i(0);
    if (has_ind) {
      ind = indirect[ind_index];
    }
    if (needs_fixed) {
      var offset = vec2i(0);
      if (matrix != f_GX_ITM_OFF && has_ind) {
        offset = f_indirect_offset(stage, ind, base, has_coord);
      }
      let next = vec2i(f_wrap(base.x, wrap_s), f_wrap(base.y, wrap_t)) + offset;
      if (add_prev) {
        texcoord += next;
      } else {
        texcoord = next;
      }
      texcoord = tev_s24_fixed_vec2i(texcoord);
    }

    var tex = vec4f(1.0);
    if (can_sample && (uses_texture || i == z_stage)) {
      let size_bias = uniform_vec4(uniform_offset(UberUniformOffsets_texSizeBias + map * 4u));
      var uv = raw_coords[coord];
      if (needs_fixed) { uv = vec2f(texcoord) / (size_bias.xy * 128.0); }
      let sampled = f_sample(map, uv, size_bias.z);
      if (i == z_stage) {
        z_sample = sampled;
      }
      tex = f_swap(sampled, cfg_u32(stage + TevStage_tevSwapTex));
    }
    let ras = f_raster(stage, in.cc0, in.cc1, ind, has_ind);
    var kc = vec3f(0.0);
    var ka = 0.0;
    if (any(color_args == vec4u(f_GX_CC_KONST))) {
      kc = f_kcolor(cfg_u32(stage + TevStage_kcSel));
    }
    if (any(alpha_args == vec4u(f_GX_CA_KONST))) {
      ka = f_kalpha(cfg_u32(stage + TevStage_kaSel));
    }
    let ca = f_color_input(color_args.x, &regs, tex, ras, kc, rgb_normalized, alpha_normalized);
    let cb = f_color_input(color_args.y, &regs, tex, ras, kc, rgb_normalized, alpha_normalized);
    let cc = f_color_input(color_args.z, &regs, tex, ras, kc, rgb_normalized, alpha_normalized);
    let cd = f_color_arg(color_args.w, &regs, tex, ras, kc);
    let aa = f_alpha_input(alpha_args.x, &regs, tex, ras, ka, alpha_normalized);
    let ab = f_alpha_input(alpha_args.y, &regs, tex, ras, ka, alpha_normalized);
    let ac = f_alpha_input(alpha_args.z, &regs, tex, ras, ka, alpha_normalized);
    let ad = f_alpha_arg(alpha_args.w, &regs, tex, ras, ka);
    // Both combiners read the pre-stage register bank. Packed alpha compares
    // consume these saved color A/B inputs, never the color result just written.
    let rgb = f_color_op(stage + TevStage_colorOp, ca, cb, cc, cd);
    let alpha = f_alpha_op(stage + TevStage_alphaOp, aa, ab, ac, ad, ca, cb);
    last_color = cfg_u32(stage + TevStage_colorOp + TevOp_outReg);
    last_alpha = cfg_u32(stage + TevStage_alphaOp + TevOp_outReg);
    regs[last_color] = vec4f(rgb, regs[last_color].a);
    regs[last_alpha].a = alpha;
    rgb_normalized = (rgb_normalized & ~(1u << last_color)) |
        select(0u, 1u << last_color, cfg_u8(stage + TevStage_colorOp + TevOp_clamp) != 0u);
    alpha_normalized = (alpha_normalized & ~(1u << last_alpha)) |
        select(0u, 1u << last_alpha, cfg_u8(stage + TevStage_alphaOp + TevOp_clamp) != 0u);
  }
  var color = vec4f(regs[last_color].rgb, regs[last_alpha].a);
  let native_depth = select(in.pos.z, 1.0 - in.pos.z, f_reversed_z);
  var fog_z = native_depth;
  var depth = in.pos.z;
  if (z_op != f_GX_ZT_DISABLE) {
    let raw = vec4u(round(clamp(z_sample, vec4f(0.0), vec4f(1.0)) * 255.0));
    var decoded: u32;
    switch ((flags >> 5u) & 3u) {
      case 0u: { decoded = raw.a; }
      case 1u: { decoded = (raw.a << 8u) | raw.r; }
      default: { decoded = (raw.r << 16u) | (raw.g << 8u) | raw.b; }
    }
    var z = uniform_u32(uniform_offset(UberUniformOffsets_ztexture)) + decoded;
    if (z_op == f_GX_ZT_ADD) {
      z += u32(clamp(native_depth, 0.0, 1.0) * 16777215.0);
    }
    let native_z = f32(z & 0x00ffffffu) / 16777215.0;
    fog_z = native_z;
    depth = select(native_z, 1.0 - native_z, f_reversed_z);
  }
  color = vec4f(f_fog(color.rgb, fog_z, in.pos.x), color.a);
  if ((rgb_normalized & (1u << last_color)) == 0u) { color = vec4f(tev_overflow_vec3f(color.rgb), color.a); }
  if ((alpha_normalized & (1u << last_alpha)) == 0u) { color.a = tev_overflow_f32(color.a); }
  if (!f_alpha_pass(color.a)) {
    discard;
  }
)";
  if (key.dstAlphaMode == DstAlphaMode::Replace) source += "  color.a = 1.0;\n";
  source += "  if ((flags & 128u) != 0u) { color.a = 1.0; }\n  var out:f_FragmentOutput;\n";
  if (key.dstAlphaMode == DstAlphaMode::DualSource) {
    source += "  out.color = vec4f(color.rgb, 1.0);\n  out.blend = color;\n";
  } else { source += "  out.color = color;\n"; }
  if (key.normalAttachment != UINT32_MAX) {
    source += R"(
  out.normal = vec4f(0.5, 0.5, 0.5, 0.0);
  if ((flags & 256u) != 0u) {
    let len_sq = dot(in.mv_nrm, in.mv_nrm);
    let unit = select(vec3f(0.0), normalize(in.mv_nrm), len_sq > 1e-10);
    out.normal = vec4f(unit * 0.5 + 0.5, select(0.0, 1.0, len_sq > 1e-10));
  }
)";
  }
  if (key.depthOutput) source += "  out.depth = depth;\n";
  source += "  return out;\n}\n";
  return source;
}
} // namespace aurora::gx
