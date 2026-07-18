from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path


def _load(name, path):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def predict(input_path: str, checkpoint_dir: str, output_dir: str, support_dir=None) -> dict:
    package, output = Path(support_dir or Path(__file__).parent), Path(output_dir)
    keypoints = _load(
        "_rsna2024_subarticular_pipeline_keypoints", package / "keypoint_engine.py"
    )
    severity = _load(
        "_rsna2024_subarticular_pipeline_severity", package / "severity_engine.py"
    )
    located = keypoints.predict(
        input_path, checkpoint_dir, str(output / "keypoints")
    )
    points = []
    for point in located["points"]:
        slug = point["point"].replace(" ", "_").replace("/", "_")
        grade = severity.predict(
            point["crop_path"],
            str(Path(checkpoint_dir) / "severity.pt"),
            str(output / "points" / slug),
        )
        points.append(
            {"point": point["point"], "keypoint": point, "severity": grade}
        )
    worst = max(points, key=lambda item: item["severity"]["grade"])
    result = {
        "task": "subarticular_stenosis_pipeline",
        "points": points,
        "maximum_grade": worst["severity"]["grade"],
        "maximum_label": worst["severity"]["label"],
        "maximum_point": worst["point"],
        "keypoint_preview": located["preview"],
        "warning": "Research use only; not a medical device.",
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


_EMBEDDED_SUPPORT = """P)h>@6aWAK2msPV@m$8k))fB;006ca000sI003)cd2nxOZggK|Zf9w3WiD`el~`ME+d2|{_pjjciE0(qcG8?Q3op?26ng^n67(FLm%tDhiB7oDZAr<oQ}n;@42QbeO4|+EL>kR}9L}BOoWIkVtd)@&v(2qEwN9E$8up;}RjG^GvW+QoCJQEa8kqiCvfr=XfBD3$GFsV)b3St%alC2j#;7=EI<HDogG^D@veso`XERUxk1Y%0uVj6n>D!k18eTiDqRFd0la>`#ORDy%PRhJ}txJ>K6VUb5#}?!&mv<_pKffFKNDPj*WhsgQajPw87v*KDG8=7ms>+tnoTl;OVm6z_@87=v;{#i<Z{Oi9RT~x;rO9QcpOi>tEv>2pcJeRKH7ZhRjND(&82nUv2dE(EnIcDs9Y-W>y|0u2nN3#8`g9qLW&Fbn=01x%1%|9myI#onS7j5KsXw!_QaogfXda9pUao#7dR2hi6IIn9KJUb{@-bxh`tBZ-0PQMdDQcZ5LBJssi;%r}6C5ZEyh)}j;n~;;A_eRf%jHfuyc5D+sFU*mR)i;l&>MEF<Mn-6m$|BqPMn^VQ4~s`Eaw4Rtr-9G<@1;Ke|-J&>BEAP$wwM3U`2Xy-vFR71rwcRm#<$313xqn>=!}n;>47d&=gL)knKZeE1Brx4j#bqah|W<oP*J5UH>ia+2S2&kjqLqCv;@m@x9D8o?#~z=OHs?Q>4NJgUx!7S8N#!g!aE6v>ymnGEG(LT`ALcM?w%<=uyERMA>=DF7N>8d}WaCpHKr@c^9yktd%0tz9a2P8mx%H?6kP)T+~6Gc;?HF(u#+N%)6F(ciB<8<sI{S=w{fjv@X>|hvdpagxOCYP`e;93QZ`)a(Naq_!ap25kZ>Byvo3&(YvdUSD!z;{Yp0Jf`saZNbXhg;EH9OYU4u!qIyhcx?ogG`dIbK68UD#PuuBfka@Ho81!Zx!hgrGDGl2|dKNNQxP;(Oi^%0Pf101-Q_!#1qXCGkfDwdeDzNw-ZUdsI#SJl%<E%_v2CjHgH9Y7*omLnXu40Nu1<799)#tKM2Tai>!B8xskGfFOh?W@XBHk#8-2-+BYYud|;k0|c4j^Yw4-rxQ*g|Pgpmb7?1RV#+jYVuEt4cwzy0>7$HXTP#3XXkhZ|3V>Irfj$=-XoOZQyUItzv)6tWh6~DUIN7T~lqFEmMV6`E90#SgZk=RBtxQK;ScvN*bkVs}t2ZlU@Z+npARBm&o`UX&cL}gk}%$B4A($#Ri^yivQ2x;2@DnYxPDeLlO5}lsv^hjS<gs7$KeI`tPG)>Aqvo47V^2oAH-hT>w4wxjaNajHa>ND4sCyXjhPQH!mo(!WL=Zb3#^InTa5(OF=#&km<z78b5{$M>_})qA>JfF#CB)p~3~t)NM}}Ra3_}h+Q&UoA97wd1Ber$kr0vtb^jTMXV86k}_>;3Y8u;i0-oTR`3^5n&dp_k_g5^lKW9+Tp<k08(Xtm#Uw0gR03hSXGvMqQWsE|pB+ov^q_gdl}tZCqAEr~SUUxs8}SLf=u}#_O3EyQo@m`%qHC2yL3MrMy0jVt9QLXqA30$l2V%4<4vKC|iY&~P0#bdI$^15zPUg~2Y4{vX;omIR-f)<)zl;hl^q2+2bCIaeO@6CPI~HWJa9+C^%(BQny&1BJEgP5xV_kq9l^^*Gt^<FeOXB#U#GL9kEdZgyU|59Vwi~p%TDmQ>CRc<48YT~YuJ?li{!Df9=Q=tqdjl$}-MVY-M`HK`BNUFH=kp)+`y<I<_WYZ(9IlL}8yl<RTMvp(4aFWr+FFadh~2?QkZl4w8bH-Z^)?o6Tx|;b-$p6e)cg>{J;0VLsEy(@M8o61AcA9EEauVKYY#M!mdh!qU#>Y}-*%oEfgIkWhP0x+>B2QHFbMLCL2~#~gl9wyM+YbTUixAQo6vW8sZ$Y*3ok9d9<+(9ywfh>{Whk+ATI7@br2YfA50$MWCxiY$H7xv`Si7Y6CLEnVj8Kn=xW#;+)&bX*ff3uK+*0GvVK$JL10Pkty>k{oeO`OOv=)vx`1|O<1SG8U6Xp0d;>oOQLbbGQ4v80hv=i2?5*|#E1d>|_&#3!<`O3C^+<<#bb4`d`sVfda()gwc}}L|e+d%j3rO3u)8F2lp~S#K&<n>pPiq4T9IrE@LzWv%LWte8#uKnFD1hts?s%vl!bjKb`tBHXM;<rsuUbYJm5^xp=zEkuz;Eb6IMmtIU#5XETkW8W#^r;1CA@O0OeXQv^-QrFmfKfw@`Jhu4HBek!N30dUE2wbd~7lBW+cL2!h;8Xa<ENCy|O=A_=NAeM2cDW1n2yz2ndIE9V9s(b*Yy1IABxoo*+X*n+W=#-%qhG96R~vfS-!)%Jsa%*5U3X=x{m!TRJebT-UD&)7?uP(<d@LxtB%Jp2lXw9#C#JAnDFxCkubkaXHe<W}13Kr*^9;(O!CtXS5l&omP?d&~ntX=q_Wj3HlHk=(7$d8F$m<{WQ5%cLW?=<HgIL{ubqzJFCr_J4{3jG3;&`R#MLJd>iwir@Zo%-a(jXZnRtVNeO3K{tY3&M^$mhpS>FBhLYC0A<mh0b+QQP7`~$1imco!(JeJtz09cELfgh`1#sj%z_F4x0^p{$*dYba)CQ+WcOb(a;h3@8_p=@*tjZujtP%omLe4_Ar#~1i?hXEcZZN}UE4O$sxMgsOJsBY1jvPVR%rAZFAAjG)?g|Y!CZdAxyRQ?$!3P1sFDY>+xU`>HaU6o7mD4!eMU{`IgUNORr_cC!w|@edo&fWJJF!fOb?)Tuz%y{~nECrbS%4n8)#`Cn`8}<VHeYfnaQxR{{l$ys4wu7@A3zHj6Z|E8JiMfjyGwfBYt~ld%hvD1Pqd!jSG)Rv;ZYhqSK*Q{4#NpNxp&tOWs&Xw>CSDE<v1z8k&9-P$S+OC5dF?+^$0KgAx|C+94esI?rMRQD1`=X1@VL~5Q=)WgaC#qv4jqCx8hB`Il17$@Kj6uv;P55O9KQH000080EejZT*ja0Vc7=&00<KR01^NI0CQz_WpZhBd0%C2XK8L_E^v8;SZ#0HI1>JzU%?}AfV7p_*lBjV27Ex;?cNnAdW)vmFF_#C5@oZ|+mTdaZ;=0fGehdlO}arFN#x9M-sfRB=O61r8C7SB6^&uCW?!y9U$NEs>J2NqYAX%1PSuUomSs(2a$QTOSZ115aL&&P(^M=?i_UdMr76=@+ZYFO>c&Z@o7$e8g|%<CsXcuw-M-Y@p7;;=9H{E9YL84>R=53<(^WORHzwPADSv$Y)JynS$(>64<2NH8!t3u%?K~~9-D?X9^*qkV+Jem$Xn2;szk2_l4{XC8pUysAz5DR#8r~yb!?rBbN|!lL7_XXK889E;t-1%l@fetMRj@RNW7F2CTxV!@q)~yj*FyyTO1V|#8XRh3w)hwO+|+7)#^9#_MN_N)QgCV|57F{%!c5cEc~t2jO5k?Peq!h1{LN(?6Q46jFQ69!{DbAfnNGQ5t0MlJ`X>MEuEF5NyQa*Aa|PHioa%+CL#A5C{(Jqwn8y4-ZrR8@1*2tGy2$fZ((H8IYk&Sj?N`0_r93LrOBS_yi1?R(zq<m>MxKO4AT9h0eCX<)4WljaS47f0!4?m<K!TLQse{{Yn?ppB;t&RJ{sw<}rU#w7eZuxi@Ahb~7jm<@WsCm({5I)dFDA>&TM}?c*b(@=?j#LJ`lokL5}rig${ea`7o7I<4}G>=O>7rLsPfdP%*lFJswg_Z89^YbG5I*KS3!$cQ;Ju;rNb(C#>iZEc7yu7Nd^U9zh*1gEOc3JqKj3+`v1#c-cHp*M}sCHed1(f;tzXGRdeEUZ)Pi@!$dv_WZCXznl(_O7FdEKvZjV&b0IxQIL3Om&f}pJje`7aOIfQ3JI$IJ3{~8u8hw82f|`-AT~X)*WcGtHQOq_Q#zmf0e5``dHQFN-cvGDX>$!B&DomKLg?y+1ogEK!(in;95$Hv$52{RQRe%GqOzTz@P+RU|HL`(5kdqK7V1@lwnM~DAL&%90%43<ZUw@6iKM1I^Qn!&Ogp2@DF?&tPMgkXJ_);x?j;X$uqY$H@pVz@(*g;81FY70^y1a}hYInMJZ$St)$PNF*y`Y(prd4%tNeC0Q5nZ>Pi|BXigVlFO*R`d(ne|WDx?P9bap(~Y?1VX<#Z=og*AUG~Cj-A>Td)vGl}E!maXD4d2&5P2WnP9;hJ?&%3!*yB8en?lqOEs0AH=t|e=7gtbkRTk3ntjpAb30i#FyYK(l%!Q*szQB?06%!RqRh$c8Xec#II-|Gtd+|LSv4s>a2r#T{{U%se{a1d1P*HlwvT$VQ}TlmO4|GkH=EQpnGm0qSE?^Pm2j{o)jH~Ii7ip6uvz*<B^z!-OJWv4EiPy(}7Q9;_`1zEX!bkSKOPpm-h;mVNq5J?5|XfGk$8;3#kk?S|_u+$ZL@>dC+#VSjLdXoDNFqpmcC5AA1%!>9os)IS2&+n==7sx8;;Fq=BaZ66{$_m0%{f8k=}eu3C5k%E<Y&ioWlJJQQ4CPxxL&*<NLLZ36{4si&dYo$O*1JoH(g`pYlhoG|$4j*DO`K`X)pr)T4|YoW|D0)iW!wH+VbR3a}RJCT7WoeJs?YK*dO_!i={1+c_n{sP1T<fdztZ+@5Hy#x_snw8R8Wj7ZIjH0Hc<5Prcn{%<eJR9qaPVoo^!tXG=nTw`}5e2^ub4}etOa?3cXT-BEmlRXKjGx$U<h|4|_n<k)A+jeX3w1BcC>E}XyePkSk9m%OlZcXu?FpOGy$}=NBni`W@e4jP0xCW2vI%=GR}$8A4(TB8$7YQD>(P$qRK%KiHU%fM9*`aB5X$8F5zs^10MF6+15_992^y#|Pi|&?V?y~wA%2=SdHk6Lo3tVfjJfsF4iuV%p({%*?dGcqo*o)^@{$T)o3GH1pnJVc@|kWJsAzO9E)#@%7l?WL_X+3iB#j-(ow~+DrC4;e{a>dP2t5BT&mJ_CyiZx(U;8K)(;?ZKW-GV4)K1ehn0wGi*dHGY+Z2uv|9n2a>dEM+0#S(PNU%j)0Fg-?Kg%5h#s-YC8p{Gzn5iV7rl^EUZTsuT4uC5r6hsaiJ78>->-hsT@>$(uhB3#dyP~?oVc$ydjPsFBV0ywiX>&)Pp(|($jX!{_eo}h>_l-7?h#CyWc0U>6BX@Ya5ke#b2MtQRb|CEbF^2-a*J`-)Or}`RbPB~zv9U0|8luZ3;P%V$I^p?KLhB&k-+gb$pSJ=>k;HzNB3^;GfFP$G_P20>^IQeL#@n!V$2EJLiF{(wqYp!nxfaWRKE-~reRpRLc81GCJhfd|nDilRs7SwzXsbt?7Ve{Xg&#c40czjz^{lRG>2|CtZZm+UK*YyCZ!8dF6G);tzM#SEAr=i9U1&ek_(aaw*+AI%F`*z2z$QH>>trpg&zwKmSBDl52;BF20dz~iVCRhsS)NcTLNnN@=mLss!al1W;HFvd7u2LbFJXP1%y`JXoF>3aZlBOUNK*rdK{rC*RTAo+bua*%x;zT@b+2uh`FR5sT{n&)EWo};=q>!Ese~*lMR2M|#*?#sfCc^;3G(m((oLmw<-`?{cU5a6PlID7hY)V?+6qms!E5Y>cdl5xo!?D7{n`HkP)h*<6ay3h000O8(nRrG#>3VW{|Ep8wio~a5&!@I0000000000q=5hc003)cd2nxOZggK|Zf9w3WiD`eP)h*<6ay3h000O8hp6&g#-Hb5*#`gs2onGR5&!@I0000000000q=7F90047kc4cyDba`K8Zf9w3WiD`eP)h{{000000ssO4fB*mhI1vB<000"""


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
