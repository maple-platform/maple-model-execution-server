from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path


def _load(name, path):
    if name in sys.modules: return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path); module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module; spec.loader.exec_module(module); return module


def predict(input_path: str, checkpoint_dir: str, output_dir: str, support_dir=None) -> dict:
    package, output = Path(support_dir or Path(__file__).parent), Path(output_dir)
    keypoints = _load("_rsna2024_foraminal_pipeline_keypoints", package / "keypoint_engine.py")
    severity = _load("_rsna2024_foraminal_pipeline_severity", package / "severity_engine.py")
    located = keypoints.predict(input_path, str(Path(checkpoint_dir) / "keypoints.pt"), str(output / "keypoints"))
    points = []
    for point in located["points"]:
        slug = point["point"].replace(" ", "_").replace("/", "_")
        grade = severity.predict(point["crop_path"], str(Path(checkpoint_dir) / "severity.pt"), str(output / "points" / slug))
        points.append({"point": point["point"], "keypoint": point, "severity": grade})
    worst = max(points, key=lambda item: item["severity"]["grade"])
    result = {"task": "neural_foraminal_narrowing_pipeline", "points": points,
              "maximum_grade": worst["severity"]["grade"], "maximum_label": worst["severity"]["label"],
              "maximum_point": worst["point"], "keypoint_preview": located["preview"],
              "warning": "Research use only; not a medical device."}
    output.mkdir(parents=True, exist_ok=True)
    (output / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


_EMBEDDED_SUPPORT = """P)h>@6aWAK2ml72@?2p78gQNm0049n000sI003)cd2nxOZggK|Zf9w3WiD`ejaXZ6+_n;a_pc!ML^ZPPb()}Yumu`7r!~+dMjD)##X_JZ?wZHCm89(51o`jh8<M*0)@kYoiy97RhI9F5R-E%Ux{^*AS+Pd-c${upBWvA&aIf}Vs~cz8#<Vq)%*w5H4m+O~?APn75AT>&Mk||f&QCl~o^N{B8<pow*IjEIlTFh)>2%xJlaomMpKY7K-${2@>Dz(%2CpMm)7RaeNz0mUAa(mv7i~SfI$cle))aR^=*{&XgTU)rZdDo{zZ&@%7(DNTwlHd7bVfE7SFd5^Nq+Uq)$ecFl70D#w^D6b-e5~s`m++HbkZu9u+yJmS=N-&7`eYZVfgFxPE{5MBU9!Hx#!5Fb^A^UoY_>Zbk8r6!!vR51M}NrtKc?g27M{xAC)a+rT(niPVtm2vh(CX#Ixo1j$Stib)h<k<LBe?s(nn^o!;KT2w2w<OXGB<L=c=Zu}ImgSILpV;4LcMiO9xINm9bjSS@$L%R3FS7wYt70*zP&5qiV^ZsW~e>)Kj5qYH0mXH-ZW7|VIWmP^LpeR%)j>i3Ty-o0J$;PL~D7SI@8{2m1KreR|A?8S>0$t(^%lKZD*2ytQBPUw&@yOixyW;<EvW{U*^pV#&B)k`=n=#6{%#Fp=1K`lGsgD{fAjqhZ&$syH0_nvVl7B5p~+P*18Q~=*i!p_)50<V3KaUU2%Xk=NcGKNu>!)=5JY6ta@Dh>?VB~btcUKygt6`2||?-KTGyhb4HN76o=h6gE;y%nob!fE8W94W)8QO#~(+Fg#6LECyBOI%D^Ur)jpbw&jp-bs6s8AUD<;^N}Rl;K|zt3N{t$hxcGvh2<EAJ^~S{_-&yV_&zWszh<8ihB=vIRR6QG?MjbgXx9^jF@sT+84;HqXXdz6A@1G7?dGX%44+@@-DC1!YBWd7hTU|cmo<QXOdPt5P4AH??KXmj9Fn6pgrDgG<1IURu=cSZKL=)WmIlP>PD5z|H#TBZ2-V|QAul+T~?|QK~sV{dY-Zp-uC9^21=Ia+2Vy)kxX{@+vV1g5x{LH=BhmhUkCP5J?KJ>UJ0*)7kw!?`9F~9HPbei4-!r57WkAf)U{$W7X`)YK~^HkTq}a8u=kDFP%+Y}bmp2gnso9nLK{*D?(=ulP1lE=OJ<!(k8>wd+ujjhsf{Q2ZbIinLTv!P6z57?C#WgT#3tFcYHtPqAuEfTCnKPu&4zQcoRk{g`LC>c_12iy2!2gsSlwI4ZWTH@^s55i-Ls-?9HNi%_=iRxog6WnEL_EwOvWJHO5sgp=%j3@6mQgeFjH0)u*Z7r*G|>Ic3+=2t<{w8xUB{P@F0UYNXDQsnJzbwGK>B#Fsm#pS>Kk@pSg^kkk;8#@>l2U=sL=ve=QYJj4~VKbyKMKeSNFU0Kb)NeB4&Ed4brUzD)E<0|%<uPYo%+6EH>it^|6fFFh|AS;EkXpC@Az1A9yWx+7#9gCb<X(BfXKf`}E=sogr*LrXF!ng9@xJ8hSzi)2<zPn87}eFHN6xpA1HiXI26?5f4AWz_^lt8sO3$`V$45T65vzhO>vpcR481<b{m0xwIu{A7BSq~DVFsr{rHL{y*~%*gM@Ac^zrMH-Q9TF(ToXW;gzk=Ox}bhA-LArHJe#4J6FwlyWjXy-#4oHeQrjXBBs#{ShS1q06KXq9AEZ|hc<B6j5iS+lnczz3Z3ld|x=?@l5sC}W;#5eel^7)KPwGlcU|myX81m>lQp%uDb*0iv+!0cZlO<h5_~w8UoQIs*uUX!*36)%W-hK!Af>h)~KD63pf8eF#|4^Po&6_fI76l(LkbtHpYqWWXu#JHbEx_J+fh5>x$flqUR){)s3viTG{%Dxg&eqerKZ&tsjIR0y9j%|7M0bnf(7kZDY0!_+;Af*+ASA~+W%OJm&Mcov=~1R^kvVnbmrSp)+=WQU1scK*Vjr%;dJIOvL?aYUa@0QgZB2IlCH!71%#g9GFVak_{D`{hAV*$Sb^pE5Aeo<$koHDYc^Cc94Ic3}jMQJ-mA4+Hri>|isyq4ECDIBicFY}IcmwJC-u4Tz<gg3+&uoF^0IEK(-_45JMd+mPvJY{S2h$gf$~Z28FiO~4&4H-T#EVPF|DH?#6ViBZ)|dt0?sSOPy*Xo7_w&lBpL;UW-joFxbnyiyx#lzs-HPT`s9H_p_~s3KH__yBhWP1!$E%K8Td%fI143974V>6It5q1IQ<5TAqKeF@dUFNad%yWIcK0tM0H`|%Bd<FkPA%aGuQXwA5eEj4d^PvxQ8jv<f1K5D$jLZ$-Acl*zfF%bZ;exAZz`RI4<$FEr&zK#N8PdC71^>~1Ek>(TO^?+X?pmA#sJoFub;}ZAK#czktEqVCdl8>KT^6_&E<zrl34k0+Y=V(K;e^85)kJ8W-N6OGfl|-9jJrbmCs{Ly}B}!J)tcvk2^qY)_ON;4-L2E`IYLKdr2BS;v)XoW_WF`8I6@e%9!ED^}0?N?|TcG52$$Pgs{fQ^@`8n{P{2x$D0|XQR000O8hp6&g#-Hb5*#`gs2onGR5&!@Ib7gjAa%psVUuAA*X>MgMaCwDTZExE+68@fF!6R^hw3XS|X?D8?d_dam-W4c%i>BBwK_Ji)WwX)SkyK)DkpF%&L+Z^<x<MOB<jip1=V3VKAL~LHRcDG7jbXB8U#>r2vDNwN4J*59D-E+w)s5AbWldvpT}!7}W|~%T&d&<dR4h%4&UHqmDbrQk7zc9d#!07}+Mb<-wQshmJ$)<PzSP^E_z(CTsOqk2k4#!txBZgSRW-afCfj={e|-GZOZZsHol5-UHzOay>+enNJT0-^YYPhXJkH44g3T3Zc$U7udjFpfY{MR(&OTke`|#-+-XmVawk*?1mpM-uubNyLFdyHox(C1U7?^Waur!Bb)7Ge5XJ~e$QGvDBLj?UwxmD#F9BN{=_!s-!)M|ak;HLmZQ>*_{aB3wF(eiD=Ow-kQROui};C9S@V&~%g&1D=DpEE}<pcexCgXO}RPPt>NBL18DCjacN!QjQarp$$N1=uj0>V>I8rdr4Td;P(f#{58T*~mNvqh(jR$n#dx>~!60fBr-5SH1S7JSx*m7PWea_?Lgby8_Kdo`gjpE&L07=<1#gqb=}PMAAIL77w>Tf|SCkgWGPKLqw9|5C(7l27h^`2c5fp!uCq<_Gqsca<jT+i~jxmHtAn4Cd<oP5^zY^5%|3BBn?RVr*}{io<!fu9I9y-oc8k%eYRXpY!^hR^3<rz$$D3+C_2CyK_IFz`8cpwL5o*YidVg*!zy^j$Xs`JgZjKl1_fWgW-Hh%bXjhqi&et<|I1(APSrw3gC-z-;$&pv4|`2jbK-JuW-Fn?L_P^*+3sbUHBh1!Sb`(6riNm3Aw5Vq#(K8S<DnFdg8XbtS*r*;&6*kvRotW+eSYhLnvt+wQRo9?_JcA}%r+awMV?iBtb))r+9MQrQ=JX#xpdMhOqj5Re5e7P9S?NU7>Vf-=tZj!s!VBBfCI2h>sAy{Tkc{tvVlgBlMpCih5c5UOw~?9$cYuoW0|mDe~rIC2&l7Cw~;4=i~vzFdriqk0vBHRQZ0UtslJw@5Tl@<*TG-dK}kq2>nFClyo@Jmce-|OK?pX;4gbWwpqY@SRdsMl2otptUALW!=y&Ra)ptkNwWYe5^-tKkU5DCn=n)L;ggKtYRNFMy5Y0&^1HWNgun<X=N5eXCIaSdJq!;LAUWQYKgv@CRqB_kQV0z@Dt#>#d#J9G8D*xhi(LenQCfL&;csv5cm*6bYHfI0Wu#5HVcq6q{>`z&CiduEVuV^4M&=fjCV~(uqtb=)7I|)jugUnocWNvSiVlcyDaOKRFI#ZU9$5O?hdu|}2()x%`iwSL>6di;)o_UNEzCAVLk(h<u%hqEI`X&$4flp-O@^4Kn%V2<4+?%+U_X?I_QC14<uT+gQernbWsSGw+C$qcAYmqQ{&~~#}#*oIG4oc~ubZ{ykdloq9w9AA!2n7I}GXZ9|<&-j{fu{fx>{(2eU?#U3n|M#IT6hA=$oaI2zVC!Q6kK0V_+CcYUS)S}0|hy$r=i)M>|zu=^jV+!%P-%YF!<<>i(o53E5Zb)XXCVMq0BP^f*YQ-9Ut9PA}=62k%1?j3hEDPjIwU{7UHx8u*6~h0>lF3rfZdNewX0A1QBDJmC{;eHx~(vqNb(eQ-o@pbFsWU8|#Zs@dyUO?=ZZXi>8MW1-}h*P2EFG1}pt%#Ir7!6jQ&9pV)5Xz0@%GpgG4OvL_}BbuY^(7OsiBD8F})d5(aSh?0rz37gWr5EI}e3Db1(3qCUfDn0G8341PA64rDM=^*dNW{mvn(T?X-#F}_E1t+r}kR9m|%H;VG&_mk*&(Zk<R2T0F8mKW(Zf1RBLit4@ewsIV{Fw!tv?2_Qx%JWx6q<ygD@!fy=Bo*w9vXM@k_um&uh5U6d%aBZnQj=UXml<v6NGygh<W?>3Fqx3jUCCIy2eALSah}hU#AoZJpV4w9yFA^Pg&hx`zRLEA=#Q{E4RATPSZ4)d(cPNA0G?b6pj%8d_KPF$>^v8QHbYAuti${kx3js%N+y828^;A%K}xHsU)DLsDw&w`|HOJfGZ{xL=GD}U~H7@`2#faS>0oXF~_I7qPoLj-%9X|^N~(qdcrwrb4Q<{D`*RiKY*-$QhNXQjW&>o8VtsEKN;a8cX+!ILL>tR4NAOrAnf)rhXTIWYPj=ErdZE(3dK&bu`s?GqRS=V_RH});rUZS>mc9XeQ(I0w*p6z#D13|UV*rPAg3Mnw{U^;Tm`?z+pu=WHG7<id}7g~4?~c-7R!G=#eTAVcV`ZEhRZ`dwOv@4^dW4hNWY9|t4Es_?xT2xA3V(gYTxnotgdP4cC0FHGk~T*#K%8xED&Q8NTNBupuy}R77ZF*Xg|~VM9$dRK-l;(p&$>yCOs$XWG$@EoIlxDhZYbB-1m9`bW6Zs=Zy<lo=_@6GuWx<0*Y(GKC2$!rdjY8)TBQzVSSy<c*wn+CcsN>pU^)@Qv--WH$vZ466&6HFaVpnJPP)8uWgw5c>@()H;y4Jz`jT5E&Qdage)pWaH>bfle2w*1^yWc^6&xDO{H|@#1)ZuRcj+pgJUL#5N`0=3QexTYwU)1u2{UC-%ULI+5Z7hO928D0~7!N00;mEp7LB_0UB_g2LJ$c5&!@a00000000000001_fdBvi0BdD=aBpdDbYEp|XK8L_E^v8JO928D0~7!N00;nwsPbIKpXXuO2LJ#F6951b00000000000001_fzAg20CQz_WpZhBd0%C2XK8L_E^v8JO9ci10000200IDj0002j4*&oF00"""


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
