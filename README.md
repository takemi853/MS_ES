## このリポジトリについて
「魔方陣の生成」を 下記論文の進化戦略のアルゴリズムを元に実装したリポジトリです

[An evolutionary algorithm for magic squares](https://ieeexplore.ieee.org/abstract/document/1299763)



## Quick Start
指定の大きさの魔方陣を 100個 出力します

`$ python main.py --n <次数> --output_path <出力先のパス>`

`例) $ python main.py --n 5 --output_path ./` ( 5次魔方陣の生成 )

出力先のファイル形式

- log_ms5.csv : 5次魔方陣 が 100個
- lof_semi_ms5.csv : 5次半魔方陣

## 引数の説明
- --n : 次数 (魔方陣の1辺の長さ)
- -sf : semi-magic_square flag (半魔方陣も生成するなら True), default = False
- --output_path : 魔方陣の出力ファイルのパス指定 (必須)

## 検証用
　論文内で実装されている数点について、実装の有無による世代数の変化の比較を行いました。

　条件を変えて 5次魔方陣を100個作り、それぞれの魔方陣の生成にかかった世代数をViolin Plotにて図示しています。

具体的な内容については以下の通り。

- sigma_t の有無による 世代数比較 (valid1_html)
- 進化戦略による 世代数の比較 (valid2_html)
- 1段階 vs 2段階 の世代数の比較 (n=4 は valid3_n4.html, n=5 は valid3_n5.html)
- 局所修正 の有無 による世代数 比較 (valid4.html)
- 入れ替え対象:a_kl または a_ij_star による 世代数比較 (valid5.html)