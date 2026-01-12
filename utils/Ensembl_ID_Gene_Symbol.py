# translate ensembl id to gene symbol
import mygene
import pandas as pd
import json
import re
from typing import Dict, Any, List
import argparse

def ensembl_to_symbol(ens_ids: List[str]) -> pd.DataFrame:
    """
    通用版：自动识别物种（ENSG/ENSMUS/ENSRNO…）→ 官方 Gene Symbol
    """
    # 前缀到 mygene 物种名
    species_map = {
        'ENSG':   'human',
        'ENSMUS': 'mouse',
        'ENSRNO': 'rat',
        'ENSDAR': 'zebrafish',
        'ENSCAF': 'dog',
    }

    # 按前缀分组，减少查询次数
    from collections import defaultdict
    grp = defaultdict(list)
    for eid in ens_ids:
        key = next((k for k in species_map if eid.upper().startswith(k)), 'unknown')
        grp[key].append(eid)

    mg = mygene.MyGeneInfo()
    out = []
    for prefix, ids in grp.items():
        if prefix == 'unknown':
            # 无法识别 → 保留原 id
            out.extend([{'Ensembl_ID': i, 'Gene_Symbol': None} for i in ids])
            continue
        res = mg.querymany(
            ids,
            scopes='ensembl.gene',
            fields='symbol',
            species=species_map[prefix],
            as_dataframe=True
        )
        tmp = res.reset_index().rename(columns={'query': 'Ensembl_ID', 'symbol': 'Gene_Symbol'})
        out.append(tmp[['Ensembl_ID', 'Gene_Symbol']])

    return pd.concat(out, ignore_index=True) if out else pd.DataFrame(columns=['Ensembl_ID', 'Gene_Symbol'])


def rest_vocab(vocab_path: str, new_vocab_path: str) -> None:
    """
    读取旧 vocab → 将 Ensembl ID 映射为 Gene Symbol → 更新/追加 → 保存
    """
    # 读取旧 vocab
    with open(vocab_path, 'r', encoding='utf-8') as f:
        vocab: Dict[str, int] = json.load(f)

    # 正则：Ensembl Gene ID
    ens_pat = re.compile(r'^(ENS[A-Z]{0,4}G\d{11})$', re.I)
    ens_keys = [k for k in vocab.keys() if ens_pat.match(k)]
    if not ens_keys:
        with open(new_vocab_path, 'w', encoding='utf-8') as f:
            json.dump(vocab, f, indent=2, ensure_ascii=False)
        return

    # 批量映射
    df = ensembl_to_symbol(ens_keys)        # 调用上面函数
    mapping = dict(zip(df['Ensembl_ID'], df['Gene_Symbol']))

    # 更新逻辑
    for ens_id in ens_keys:
        symbol = mapping.get(ens_id)
        if pd.isna(symbol) or symbol == ens_id:
            continue

        old_id = vocab[ens_id]

        if symbol in vocab:
            # Gene Symbol 已存在 → 把 ens_id 的 id 改成 symbol 的 id
            vocab[ens_id] = vocab[symbol]
        else:
            # Gene Symbol 不存在 → 追加；ens_id 指向新 id
            new_id = max(vocab.values()) + 1
            vocab[symbol] = new_id
            vocab[ens_id] = new_id

    # 写出
    with open(new_vocab_path, 'w', encoding='utf-8') as f:
        json.dump(vocab, f, indent=2, ensure_ascii=False)
# ===== demo =====
if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="updata vocab translate ensembl id to gene symbol"
    )
    parser.add_argument('-v', '--vocab-path',      default='project1/spatial_data/spatial_data/vocab.json',help='Path to vocab JSON file')
    parser.add_argument('-n', '--new-vocab-path',  default='project1/spatial_data/spatial_data/new_vocab.json',help='path to new vocab JSON file')
    parser.add_argument('-s', '--seed',            type=int, default=0,                                    help='Random seed')
    args = parser.parse_args()
    rest_vocab(args.vocab_path,args.new_vocab_path)
    