import timm
import torch
import torch.nn as nn


def list_proj_modules(model):
    """
    列出 model 內所有 **名稱包含 'proj'** 的模組（通常是 Linear/Conv 的投影層）。
    僅比對模組名稱（named_modules 的 name 字串），不管類型，只要名稱含 'proj' 就列。
    回傳 list[dict]: [{"name": str, "type": str, "weight_shape": tuple|None}]
    """
    out = []
    for name, mod in model.named_modules():
        # 只要名稱中含 'proj'（大小寫敏感，timm 都是小寫）
        if "proj" in name.split(".")[-1] or name.endswith(".proj") or "proj" in name:
            wshape = (
                tuple(mod.weight.shape)
                if hasattr(mod, "weight") and mod.weight is not None
                else None
            )
            out.append(
                {"name": name, "type": type(mod).__name__, "weight_shape": wshape}
            )
    # 去掉頂層本體（空字串名稱）與重複（理論上不會有）
    out = [x for x in out if x["name"]]
    # 依名稱排序
    out.sort(key=lambda x: x["name"])
    return out


def pretty_print_proj(listing):
    if not listing:
        print("No modules with 'proj' in their name were found.")
        return
    w1 = max(len(x["name"]) for x in listing)
    w2 = max(len(x["type"]) for x in listing)
    header = f'{"name".ljust(w1)}  {"type".ljust(w2)}  weight_shape'
    print(header)
    print("-" * len(header))
    for x in listing:
        shp = str(x["weight_shape"])
        print(f'{x["name"].ljust(w1)}  {x["type"].ljust(w2)}  {shp}')


if __name__ == "__main__":
    model = timm.create_model("deit_tiny_patch16_224", pretrained=True)
    proj_list = list_proj_modules(model)
    pretty_print_proj(proj_list)

    # proj_list 就是你要的清單；你也可以這樣拿出 name 與 shape 做後續處理：
    # for p in proj_list:
    #     print(p["name"], p["weight_shape"])
