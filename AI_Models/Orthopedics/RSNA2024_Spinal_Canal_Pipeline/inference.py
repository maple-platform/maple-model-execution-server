"""End-to-end sagittal lumbar localization and spinal-canal grading."""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path


def _load(name: str, path: Path):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def predict(input_path: str, checkpoint_dir: str, output_dir: str, support_dir=None) -> dict:
    package = Path(support_dir or Path(__file__).parent)
    output = Path(output_dir)
    keypoints = _load("_rsna2024_pipeline_keypoints", package / "keypoint_engine.py")
    severity = _load("_rsna2024_pipeline_severity", package / "severity_engine.py")
    keypoint_result = keypoints.predict(
        input_path,
        str(Path(checkpoint_dir) / "keypoints.pt"),
        str(output / "keypoints"),
    )
    levels = []
    for point in keypoint_result["points"]:
        slug = point["level"].lower().replace("/", "_")
        result = severity.predict(
            point["crop_path"],
            str(Path(checkpoint_dir) / "severity.pt"),
            str(output / "levels" / slug),
        )
        levels.append({"level": point["level"], "keypoint": point, "severity": result})
    worst = max(levels, key=lambda item: item["severity"]["grade"])
    result = {
        "task": "spinal_canal_stenosis_pipeline",
        "levels": levels,
        "maximum_grade": worst["severity"]["grade"],
        "maximum_label": worst["severity"]["label"],
        "maximum_level": worst["level"],
        "keypoint_preview": keypoint_result["preview"],
        "warning": "Research use only; not a medical device.",
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


_EMBEDDED_SUPPORT = """P)h>@6aWAK2ml72@?5>&oAL?>006ub000sI003)cd2nxOZggK|Zf9w3WiD`ejaXZA+qM#Z_pd<l60o5uOOA3Jb<~;G^*Pmyav9sx7mtPmlaP&?w}PN8CX@eucCp|^igI%FAc4T{VzImU_M<rG>n6XKhGptmW$a0{RjCWhbg@-N6^UY7Q|3&vT0UrNWyWr2lbhSsHTz?A`SCrgmC>q>IOiu$Fpjs4ZH$Uzrt_*amdT<ht+cu<>XQ>c_jg?uBwtDUnCbgY_!B(#Qbm(jEt54Xs&1-kQ=OD~_h@xKj7yU|lF?7Ab!Tvu%Lf(G&mTs<kbskY)-^cQ30fnIdRv+tygXT7{dKjzVM`|XdOBIpc*yvAHd&v$!TDtU#tq&~)^j(QPj03>IEgQRyZrMC)V_QLBdOZ5xB!bX)1Q?{rIod^0XzK_4ipusG)A_I69%73?^ITUq8EzXMC>FYS=+W!0(7=nDed_z*w^ucpP5@04+^4fO}8JocdhC~X6hy@D+TUOqqAVo#OvjcR9A&nCQ+3I^|M|*D_=tPs2?7|3Gl8$k;3Xs2{IfqF%8+fcfo<fAe&^m5?+j*qNRXMST1+M**hiMQ+4__fE{5(CiIql>*J@#(w4chMkmhC$|$NgObrj%a>@Anj~_l>{(1fJ{neCH$oD*&!jAOd*1$kx3MTr<&gS!Al!pdx$VJelI5A}<G#1!ml-Q8%78KTywf(5BWTJ})7(hwHdF~g$g?DcuuE%P_tHf=u9q&LsmzCHzN+Eib+13l}#Pn^*OxYBv@cTiK2O|?U3r0%q|Dn{5lq#8~D)qUSDTM%XF34KwtwI86X6I*ifj5BFq72Ua5qdy7?*jIkby_sl_ERlsL)fwUofmg~PJ2_Q-KiGCTxTFW6yJ5?yFq+6xZuP$<Jy2T%i3j0^tK$xL<lgnfF1^sQ5Z%cX0taTgRj7&Q3Q8O=2Zq^jecLPS0Ao^yQaAGm6eyN%0%+0k|&4CF{0&y)z$=y?mir>0&p{ep`pyZ%QSs=pg^Y@CLyE_9V4i%^l<MG%P!8!#NDqYPpXEK&(2lU1w@V#qoh^d{UOx7tvh}%lc)Q#P<#_Iyst*;LZ!=p%dCbH0^pC6OxCrkm(zL3QV68mKQ>@@c@|CQK2ZJE&fU1SxG%H_E5^Gy3BDomR6T2GYuz#FQShWmCC7l$G(JYUj^(q|nY;%~3m9O5Vuxpr)U(V)5ZO|YDO&qZu(72$wQv-Ukc~BekS!_=h?zU3E~;i!79l%UEFmi!i*l(pR6U3@sP7TR^>ADT_pk$x%#u%A0CmL48akx{T<8^|hpfC8{O2f5avt;`4IQ6!Mnxz!MDH(|HR{Tk(g?o7O<3O4mfb607+_8%5dD@VWnrNp;AV`8$Cz;V4EQF1uHAWRfSy44rNVj7prf<yUO107dx{aBNC4|uW-`A|CDT>`C`0%Jj-pioB?T<#1CS(TI^fG>;lgn@YVRmc+u;be>-N}mQOQ&xzBuKmew39W=kOWD$?DgD{RI8!f5pwmA%lvkDSVIeSuyob13^l4t_tKeU5FgUs8|=NlXqxs@O<zK*n_~X1aw$*j$DB1V9>YEgTCc~xKAwt7uYyNI<N(CN}j|WnYe@P)NVuJz#$%R-xIj_a;NL%=`<MKreoIv_B)P*q}LNSywW|1i!Mvxb+sDcXJ`?7S~cRZ1L!egxP!(x4+s??0|0k{+d!^my}UJz3c~N%yDe`a1(2!li+t$IMw~@+x4}8I$$mW2+>DU!MMEJIKK1<Eg+>HJ!6apAQe6NLVzw!8{npf^gQzL$e;TF0fwM7m1yQXZ@=~Y5cj!G^qo)W^5J1l{bm4j&7POdS*$|^pz~dl=FFLiz4}D#nZK(8}453noIRi!oeSi0u<OL@HzA#zyz4a)6f^WF)5DrVZ37LY@vT^xDsiIqpnRU08LRvm6lgai-HG5z8>-BEB*#r?F6R26iumAIVhe7aG(HX%vj0nGkx88*@D-CW9EMPuI762FS42co^x|!+Q{&7Zg_()5=uzS2fHhj@haJ*pd%?+AnS5s+78{(b&3^bcwX2Nm$yhJ@J8>Ec%7BCosbU`%=${~P)-xNRXG4ZJrr_>+F6ACPuM;^!jiPw$vXHFlHsKd^H%(!=eBl9C}0k4_F<$Zd)=#<+nC}1qa>D0eVEcWiDL(=$|gEaTr+xVdqV@D!cRSHf=ZwFmP4Lbq+0kZ{?+!@<jj6UI;$BpOpqhoeN2ybI|KPGI;zqld*67-^syN|FV;QFh{tB}2lUj<*Is(9evWvV@XckU1196AVDQo%<xbQ;lnF*t^oAW|6w4Ioz_n}$+t=?5d|p5O<}K^M-JuKW(vt=d`Cea-$wx)(v%756bEy3k#q_p<P1?+HKL8R83X@Z0rW?B4c3Vs~V<?N{3)tE1HM1(o^{DebvLTsShpiDvZq_nALn97pU77bf1dpTW#90Z^U+@QGt;eb%_+fTJw#3xDmOf-z2ZWjA0kaL{M1UWUr@l2jyPwT{@<O@?qkWQW7qpTHlUxj&&|mz=o!6RPHeXUO5<jJ8GTZ~lvVf4v&ZdhLZTk?{)o>_f2U<4Ed78YG{%6_5iZq2aOySXfzP?JthpN|qx5N5bVgXXLl1;E5qfbm$2G@Pg~}Vt^dRuF!ENN}+$&f@DHhIE7u#pzNWumw?p|OWxS+=>^|=H6Zzu{{m1;0|XQR000O8hp6&g#-Hb5*#`gs2onGR5&!@Ib7gjAa%psVUuAA*X>MgMaCwDTZExE+68@fF!6R^hw3XS|X?D8?d_dam-W4c%i>BBwK_Ji)WwX)SkyK)DkpF%&L+Z^<x<MOB<jip1=V3VKAL~LHRcDG7jbXB8U#>r2vDNwN4J*59D-E+w)s5AbWldvpT}!7}W|~%T&d&<dR4h%4&UHqmDbrQk7zc9d#!07}+Mb<-wQshmJ$)<PzSP^E_z(CTsOqk2k4#!txBZgSRW-afCfj={e|-GZOZZsHol5-UHzOay>+enNJT0-^YYPhXJkH44g3T3Zc$U7udjFpfY{MR(&OTke`|#-+-XmVawk*?1mpM-uubNyLFdyHox(C1U7?^Waur!Bb)7Ge5XJ~e$QGvDBLj?UwxmD#F9BN{=_!s-!)M|ak;HLmZQ>*_{aB3wF(eiD=Ow-kQROui};C9S@V&~%g&1D=DpEE}<pcexCgXO}RPPt>NBL18DCjacN!QjQarp$$N1=uj0>V>I8rdr4Td;P(f#{58T*~mNvqh(jR$n#dx>~!60fBr-5SH1S7JSx*m7PWea_?Lgby8_Kdo`gjpE&L07=<1#gqb=}PMAAIL77w>Tf|SCkgWGPKLqw9|5C(7l27h^`2c5fp!uCq<_Gqsca<jT+i~jxmHtAn4Cd<oP5^zY^5%|3BBn?RVr*}{io<!fu9I9y-oc8k%eYRXpY!^hR^3<rz$$D3+C_2CyK_IFz`8cpwL5o*YidVg*!zy^j$Xs`JgZjKl1_fWgW-Hh%bXjhqi&et<|I1(APSrw3gC-z-;$&pv4|`2jbK-JuW-Fn?L_P^*+3sbUHBh1!Sb`(6riNm3Aw5Vq#(K8S<DnFdg8XbtS*r*;&6*kvRotW+eSYhLnvt+wQRo9?_JcA}%r+awMV?iBtb))r+9MQrQ=JX#xpdMhOqj5Re5e7P9S?NU7>Vf-=tZj!s!VBBfCI2h>sAy{Tkc{tvVlgBlMpCih5c5UOw~?9$cYuoW0|mDe~rIC2&l7Cw~;4=i~vzFdriqk0vBHRQZ0UtslJw@5Tl@<*TG-dK}kq2>nFClyo@Jmce-|OK?pX;4gbWwpqY@SRdsMl2otptUALW!=y&Ra)ptkNwWYe5^-tKkU5DCn=n)L;ggKtYRNFMy5Y0&^1HWNgun<X=N5eXCIaSdJq!;LAUWQYKgv@CRqB_kQV0z@Dt#>#d#J9G8D*xhi(LenQCfL&;csv5cm*6bYHfI0Wu#5HVcq6q{>`z&CiduEVuV^4M&=fjCV~(uqtb=)7I|)jugUnocWNvSiVlcyDaOKRFI#ZU9$5O?hdu|}2()x%`iwSL>6di;)o_UNEzCAVLk(h<u%hqEI`X&$4flp-O@^4Kn%V2<4+?%+U_X?I_QC14<uT+gQernbWsSGw+C$qcAYmqQ{&~~#}#*oIG4oc~ubZ{ykdloq9w9AA!2n7I}GXZ9|<&-j{fu{fx>{(2eU?#U3n|M#IT6hA=$oaI2zVC!Q6kK0V_+CcYUS)S}0|hy$r=i)M>|zu=^jV+!%P-%YF!<<>i(o53E5Zb)XXCVMq0BP^f*YQ-9Ut9PA}=62k%1?j3hEDPjIwU{7UHx8u*6~h0>lF3rfZdNewX0A1QBDJmC{;eHx~(vqNb(eQ-o@pbFsWU8|#Zs@dyUO?=ZZXi>8MW1-}h*P2EFG1}pt%#Ir7!6jQ&9pV)5Xz0@%GpgG4OvL_}BbuY^(7OsiBD8F})d5(aSh?0rz37gWr5EI}e3Db1(3qCUfDn0G8341PA64rDM=^*dNW{mvn(T?X-#F}_E1t+r}kR9m|%H;VG&_mk*&(Zk<R2T0F8mKW(Zf1RBLit4@ewsIV{Fw!tv?2_Qx%JWx6q<ygD@!fy=Bo*w9vXM@k_um&uh5U6d%aBZnQj=UXml<v6NGygh<W?>3Fqx3jUCCIy2eALSah}hU#AoZJpV4w9yFA^Pg&hx`zRLEA=#Q{E4RATPSZ4)d(cPNA0G?b6pj%8d_KPF$>^v8QHbYAuti${kx3js%N+y828^;A%K}xHsU)DLsDw&w`|HOJfGZ{xL=GD}U~H7@`2#faS>0oXF~_I7qPoLj-%9X|^N~(qdcrwrb4Q<{D`*RiKY*-$QhNXQjW&>o8VtsEKN;a8cX+!ILL>tR4NAOrAnf)rhXTIWYPj=ErdZE(3dK&bu`s?GqRS=V_RH});rUZS>mc9XeQ(I0w*p6z#D13|UV*rPAg3Mnw{U^;Tm`?z+pu=WHG7<id}7g~4?~c-7R!G=#eTAVcV`ZEhRZ`dwOv@4^dW4hNWY9|t4Es_?xT2xA3V(gYTxnotgdP4cC0FHGk~T*#K%8xED&Q8NTNBupuy}R77ZF*Xg|~VM9$dRK-l;(p&$>yCOs$XWG$@EoIlxDhZYbB-1m9`bW6Zs=Zy<lo=_@6GuWx<0*Y(GKC2$!rdjY8)TBQzVSSy<c*wn+CcsN>pU^)@Qv--WH$vZ466&6HFaVpnJPP)8uWgw5c>@()H;y4Jz`jT5E&Qdage)pWaH>bfle2w*1^yWc^6&xDO{H|@#1)ZuRcj+pgJUL#5N`0=3QexTYwU)1u2{UC-%ULI+5Z7hO928D0~7!N00;mEp7LD1-<$FZ2mk=Q6951b00000000000001_fdBvi0BdD=aBpdDbYEp|XK8L_E^v8JO928D0~7!N00;nwsPbIKpXXuO2LJ#F6951b00000000000001_fjS5P0CQz_WpZhBd0%C2XK8L_E^v8JO9ci10000200IDj0000(5C8xG00"""


def _maple_materialize_support(directory):
    import base64
    import io
    import zipfile

    target = Path(directory) / "_support"
    target.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(io.BytesIO(base64.b85decode(_EMBEDDED_SUPPORT))) as bundle:
        for member in bundle.infolist():
            destination = (target / member.filename).resolve()
            if target.resolve() not in destination.parents and destination != target.resolve():
                raise ValueError("Invalid embedded support path")
        bundle.extractall(target)
    return target


def _maple_split_input(input_data):
    if isinstance(input_data, (str, Path)):
        return str(input_data), {}
    if not isinstance(input_data, dict):
        raise TypeError("input_data must be a path or a dictionary")
    input_path = next(
        (
            input_data[key]
            for key in ("image_path", "input_path", "series_path", "volume_path")
            if input_data.get(key)
        ),
        None,
    )
    if input_path is None:
        raise ValueError(
            "input_data dictionary requires image_path, input_path, "
            "series_path, or volume_path"
        )
    params = {
        key: value for key, value in input_data.items() if not key.endswith("_path")
    }
    return str(input_path), params


def _maple_model_root(model_path):
    path = Path(model_path).expanduser().resolve()
    return path if path.is_dir() else path.parent


def _maple_checkpoint(model_path, filename):
    path = Path(model_path).expanduser().resolve()
    if path.is_file() and path.name == filename:
        return path
    candidate = (path if path.is_dir() else path.parent) / filename
    if not candidate.is_file():
        raise FileNotFoundError(f"Required checkpoint not found: {candidate}")
    return candidate


def _maple_public_value(value):
    import numpy as np

    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        public = {}
        for key, item in value.items():
            if (
                key in {"ensemble_members", "fold_predictions_months", "runtime_log_tail"}
                or key == "preview"
                or key.endswith(("_preview", "_b64", "_file", "_path"))
            ):
                continue
            public[key] = _maple_public_value(item)
        return public
    if isinstance(value, (list, tuple)):
        return [_maple_public_value(item) for item in value]
    return value


def _maple_finish(result, image_paths, image_roles, summary):
    import numpy as np
    from PIL import Image

    images = [
        np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8)
        for path in image_paths
        if path and Path(path).is_file()
    ]
    if not images:
        raise RuntimeError("Inference did not produce a display image")
    prediction = _maple_public_value(result)
    prediction.update(_maple_public_value(summary))
    prediction["image_roles"] = list(image_roles)
    display = images[0] if len(images) == 1 else images
    return display, [prediction]


def _maple_series(input_path):
    source = Path(input_path)
    return str(source.parent if source.is_file() else source)


def main(input_data, model_path: str):
    import tempfile

    input_path, _ = _maple_split_input(input_data)
    with tempfile.TemporaryDirectory(prefix="maple_rsna_pipeline_") as output_dir:
        support = _maple_materialize_support(output_dir)
        result = predict(
            _maple_series(input_path), str(_maple_model_root(model_path)), output_dir, support
        )
        rows = result.get("levels", result.get("points", []))
        images = [result["keypoint_preview"]]
        images.extend(row["severity"]["preview"] for row in rows)
        roles = ["keypoint_overlay"]
        roles.extend(
            f"{row.get('level', row.get('point', 'region'))}_severity_overlay"
            for row in rows
        )
        return _maple_finish(
            result,
            images,
            roles,
            {"pred": result["maximum_grade"], "pred_name": result["maximum_label"]},
        )
