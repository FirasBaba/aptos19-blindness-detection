import argparse
from collections import defaultdict, Counter
import random

import pandas as pd
from tqdm import tqdm
ch="""
../input/train_images/00cb6555d108.png
../input/train_images/ca25745942b0.png

../input/train_images/012a242ac6ff.png
../input/train_images/1f07dae3cadb.png

../input/train_images/0161338f53cc.png
../input/train_images/cac40227d3b2.png

../input/train_images/0243404e8a00.png
../input/train_images/3ddb86eb530e.png
../input/train_images/3b018e8b7303.png

../input/train_images/026dcd9af143.png
../input/train_images/98e8adcf085c.png

../input/train_images/034cb07a550f.png
../input/train_images/c8d2d32f7f29.png

../input/train_images/04ac765f91a1.png
../input/train_images/3044022c6969.png

../input/train_images/05a5183c92d0.png
../input/train_images/63a03880939c.png

../input/train_images/07419eddd6be.png
../input/train_images/81b0a2651c45.png

../input/train_images/1b862fb6f65d.png
../input/train_images/0a4e1a29ffff.png

../input/train_images/0ac436400db4.png
../input/train_images/fda39982a810.png

../input/train_images/0c7e82daf5a0.png
../input/train_images/3e86335bc2fd.png

../input/train_images/0cb14014117d.png
../input/train_images/4a44cc840ebe.png

../input/train_images/0dce95217626.png
../input/train_images/94372043d55b.png

../input/train_images/1006345f70b7.png
../input/train_images/435d900fa7b2.png

../input/train_images/111898ab463d.png
../input/train_images/33105f9b3a04.png

../input/train_images/11242a67122d.png
../input/train_images/65c958379680.png

../input/train_images/12e3f5f2cb17.png
../input/train_images/d567a1a22d33.png

../input/train_images/135575dc57c9.png
../input/train_images/2c2aa057afc5.png

../input/train_images/1411c8ab7161.png
../input/train_images/8acffaf1f4b9.png

../input/train_images/144b01e7b993.png
../input/train_images/1e143fa3de57.png

../input/train_images/14515b8f19b6.png
../input/train_images/ba2624883599.png

../input/train_images/14e3f84445f7.png
../input/train_images/f0f89314e860.png

../input/train_images/155e2df6bfcf.png
../input/train_images/415f2d2bd2a1.png

../input/train_images/1632c4311fc9.png
../input/train_images/a75bab2463d4.png

../input/train_images/1638404f385c.png
../input/train_images/576e189d23d4.png

../input/train_images/16ce555748d8.png
../input/train_images/4d9fc85a8259.png

../input/train_images/19722bff5a09.png
../input/train_images/19e350c7c83c.png

../input/train_images/1a1b4b2450ca.png
../input/train_images/92b0d27fc0ec.png

../input/train_images/1ae8c165fd53.png
../input/train_images/5a36cea278ae.png

../input/train_images/d6b109c82067.png
../input/train_images/81914ceb4e74.png
../input/train_images/1b398c0494d1.png

../input/train_images/1b4625877527.png
../input/train_images/86b3a7929bec.png

../input/train_images/1c4d87baaffc.png
../input/train_images/35aa7f5c2ec0.png

../input/train_images/1c5e6cdc7ee1.png
../input/train_images/3c53198519f7.png

../input/train_images/1c6d119c3d70.png
../input/train_images/a56230242a95.png

../input/train_images/1c9c583c10bf.png
../input/train_images/ea15a290eb96.png

../input/train_images/1cb814ed6332.png
../input/train_images/a7b0d0c51731.png

../input/train_images/1dfbede13143.png
../input/train_images/38fe9f854046.png

../input/train_images/1e8a1fdee5b9.png
../input/train_images/a47432cd41e7.png
../input/train_images/b8ebedd382de.png

../input/train_images/1e9224ccca95.png
../input/train_images/f6f3ea0d2693.png

../input/train_images/1ee1eb7943db.png
../input/train_images/c2d2b4f536da.png

../input/train_images/22895c89792f.png
../input/train_images/bd5013540a13.png

../input/train_images/23d7ca170bdb.png
../input/train_images/ea9e0fb6fb0b.png

../input/train_images/26e231747848.png
../input/train_images/36ec36c301c1.png

../input/train_images/59bd19c1c5bb.png
../input/train_images/26fc2358a38d.png

../input/train_images/278aa860dffd.png
../input/train_images/f066db7a2efe.png

../input/train_images/2923971566fe.png
../input/train_images/badb5ff8d3c7.png

../input/train_images/7e6e90a93aa5.png
../input/train_images/29f44aea93a4.png

../input/train_images/2a3a1ed1c285.png
../input/train_images/2b21d293fdf2.png

../input/train_images/9a94e0316ee3.png
../input/train_images/2b48daf24be0.png

../input/train_images/2cceb07ff706.png
../input/train_images/8a759f94613a.png

../input/train_images/2df07eb5779f.png
../input/train_images/b91ef82e723a.png

../input/train_images/2f284b6a1940.png
../input/train_images/bb5083fae98f.png

../input/train_images/2f7789c1e046.png
../input/train_images/a8e88d4891c4.png

../input/train_images/30cab14951ac.png
../input/train_images/c546670d9684.png

../input/train_images/4ccfa0b4e96c.png
../input/train_images/33778d136069.png

../input/train_images/36041171f441.png
../input/train_images/595446774178.png

../input/train_images/36677b70b1ef.png
../input/train_images/7bf981d9c7fe.png

../input/train_images/38487e1a5b1f.png
../input/train_images/b376def52ccc.png

../input/train_images/3a1d3ce00f0c.png
../input/train_images/a19507501b40.png

../input/train_images/3b4a5fcbe5e0.png
../input/train_images/3ca637fddd56.png

../input/train_images/3cd801ffdbf0.png
../input/train_images/7525ebb3434d.png

../input/train_images/3dbfbc11e105.png
../input/train_images/d0079cc188e9.png

../input/train_images/3ee4841936ef.png
../input/train_images/7005be54cab1.png

../input/train_images/3f44d749cd0b.png
../input/train_images/f9e1c439d4c8.png

../input/train_images/3fd7df6099e3.png
../input/train_images/a3b2e93d058b.png

../input/train_images/40e9b5630438.png
../input/train_images/77a9538b8362.png

../input/train_images/42985aa2e32f.png
../input/train_images/6165081b9021.png

../input/train_images/42a850acd2ac.png
../input/train_images/51131b48f9d4.png
../input/train_images/8cb6b0efaaac.png

../input/train_images/435414ccccf7.png
../input/train_images/9f1b14dfa14c.png

../input/train_images/43fb6eda9b97.png
../input/train_images/e4e343eaae2a.png

../input/train_images/4478b870e549.png
../input/train_images/89ee1fa16f90.png

../input/train_images/46cdc8b685bd.png
../input/train_images/e4151feb8443.png

../input/train_images/48c49f662f7d.png
../input/train_images/6cb98da77e3e.png

../input/train_images/bf7b4eae7ad0.png
../input/train_images/496155f71d0a.png

../input/train_images/4ce74e5eb51d.png
../input/train_images/c68dfa021d62.png

../input/train_images/668a319c2d23.png
../input/train_images/4d167ca69ea8.png

../input/train_images/4d7d6928534a.png
../input/train_images/94b1d8ad35ec.png

../input/train_images/4fecf87184e6.png
../input/train_images/9bf060db8376.png
../input/train_images/f7edc074f06b.png

../input/train_images/521d3e264d71.png
../input/train_images/fe0fc67c7980.png

../input/train_images/530d78467615.png
../input/train_images/c1c8550508e0.png

../input/train_images/5b76117c4bcb.png
../input/train_images/e037643244b7.png

../input/train_images/5dc23e440de3.png
../input/train_images/f4d3777f2710.png

../input/train_images/5e7db41b3bee.png
../input/train_images/a1b12fdce6c3.png

../input/train_images/5eb311bcb5f9.png
../input/train_images/a9e984b57556.png

../input/train_images/60f15dd68d30.png
../input/train_images/fcc6aa6755e6.png
../input/train_images/772af553b8b7.png

../input/train_images/6253f23229b1.png
../input/train_images/76cfe8967f7d.png

../input/train_images/64678182d8a8.png
../input/train_images/6b00cb764237.png

../input/train_images/65e51e18242b.png
../input/train_images/cc12453ea915.png

../input/train_images/68332fdcaa70.png
../input/train_images/d801c0a66738.png

../input/train_images/6c3745a222da.png
../input/train_images/eadc57064154.png

../input/train_images/6e92b1c5ac8e.png
../input/train_images/9c52b87d01f1.png

../input/train_images/71c1a3cdbe47.png
../input/train_images/79ce83c07588.png

../input/train_images/7550966ef777.png
../input/train_images/8d7bb0649a02.png

../input/train_images/75a7bc945b7d.png
../input/train_images/98104c8c67eb.png

../input/train_images/de55ed25e0e8.png
../input/train_images/76095c338728.png
../input/train_images/84b79243e430.png
../input/train_images/bd34a0639575.png

../input/train_images/7877be80901c.png
../input/train_images/ee2c2a5f7d0e.png

../input/train_images/7a0cff4c24b2.png
../input/train_images/86baef833ae0.png

../input/train_images/7a3ea1779b13.png
../input/train_images/a8582e346df0.png
../input/train_images/c027e5482e8c.png
../input/train_images/ca6842bfcbc9.png

../input/train_images/7b691d9ced34.png
../input/train_images/d51c2153d151.png

../input/train_images/7d261f986bef.png
../input/train_images/91cbe1c775ef.png

../input/train_images/7e160c8b611e.png
../input/train_images/91b6ebaa3678.png

../input/train_images/7e980424868e.png
../input/train_images/b10fca20c885.png

../input/train_images/80964d8e0863.png
../input/train_images/ab50123abadb.png

../input/train_images/80d24897669f.png
../input/train_images/fea14b3d44b0.png

../input/train_images/80feb1f7ca5e.png
../input/train_images/878a3a097436.png

../input/train_images/f0098e9d4aee.png
../input/train_images/8273fdb4405e.png

../input/train_images/840527bc6628.png
../input/train_images/857002ed4e49.png

../input/train_images/8446826853d0.png
../input/train_images/8ef2eb8c51c4.png

../input/train_images/bfefa7344e7d.png
../input/train_images/d85ea1220a03.png
../input/train_images/8688f3d0fcaf.png

../input/train_images/8fc09fecd22f.png
../input/train_images/d1cad012a254.png

../input/train_images/906d02fb822d.png
../input/train_images/a4012932e18d.png

../input/train_images/98f7136d2e7a.png
../input/train_images/e740af6ac6ea.png

../input/train_images/9a3c03a5ad0f.png
../input/train_images/f03d3c4ce7fb.png

../input/train_images/9b32e8ef0ca0.png
../input/train_images/a15652b22ab8.png

../input/train_images/9b418ce42c13.png
../input/train_images/be161517d3ac.png

../input/train_images/9b7b6e4db1d5.png
../input/train_images/9f4132bd6ed6.png

../input/train_images/9c5dd3612f0c.png
../input/train_images/c9f0dc2c8b43.png

../input/train_images/9e3510963315.png
../input/train_images/b187b3c93afb.png

../input/train_images/9f1efb799b7b.png
../input/train_images/cd4e7f9fa1a9.png

../input/train_images/a505981d1cab.png
../input/train_images/d994203deb64.png

../input/train_images/a8b637abd96b.png
../input/train_images/e2c3b037413b.png

../input/train_images/aca88f566228.png
../input/train_images/c05b7b4c22fe.png

../input/train_images/aeed1f251ceb.png
../input/train_images/c0e509786f7f.png

../input/train_images/b019a49787c1.png
../input/train_images/e1fb532f55df.png

../input/train_images/b06dabab4f09.png
../input/train_images/d144144a2f3f.png

../input/train_images/b13d72ceea26.png
../input/train_images/da0a1043abf7.png

../input/train_images/b8ac328009e0.png
../input/train_images/ff0740cb484a.png

../input/train_images/b9127e38d9b9.png
../input/train_images/e39b627cf648.png

../input/train_images/ba735b286d62.png
../input/train_images/ed3a0fc5b546.png

../input/train_images/bacfb1029f6b.png
../input/train_images/e12d41e7b221.png

../input/train_images/bb7e0a2544cd.png
../input/train_images/e76a9cbb2a8c.png

../input/train_images/bcdc8db5423b.png
../input/train_images/f920ccd926db.png

../input/train_images/c9e697117f3f.png
../input/train_images/e135d7ba9a0e.png

../input/train_images/ca0f1a17c8e5.png
../input/train_images/ea05c22d92e9.png

../input/train_images/ca891d37a43c.png
../input/train_images/dc3c0d8ee20b.png

../input/train_images/cd3fd04d72f5.png
../input/train_images/d81b6ed83bc2.png

../input/train_images/cd93a472e5cd.png
../input/train_images/d035c2bd9104.png

../input/train_images/ce887b196c23.png
../input/train_images/e7a7187066ad.png

../input/train_images/d28bd830c171.png
../input/train_images/f9ecf1795804.png

../input/train_images/d51b3fe0fa1b.png
../input/train_images/df4913ca3712.png

../input/train_images/e8d1c6c07cf2.png
../input/train_images/f23902998c21.png

../input/train_images/f1a761c68559.png
../input/train_images/ff52392372d3.png"""

ch1 = ch.split("\n")
new_ch = []
for elem in ch1 : 
    if elem !='':
        new_ch.append(elem.split("/")[3][:-4])





def make_folds(n_folds):
    df = pd.read_csv('../input/train.csv')
    id2keep = (set(df.id_code)-set(new_ch))
    df = df[df.id_code.isin(id2keep)].reset_index(drop=True)
#     import pdb ; pdb.set_trace()
    df['diagnosis']=df['diagnosis'].astype("str")
    cls_counts = Counter(cls for classes in df['diagnosis'].str.split()
                         for cls in classes)
    fold_cls_counts = defaultdict(int)
    folds = [-1] * len(df)
    for item in tqdm(df.sample(frac=1, random_state=42).itertuples(),
                          total=len(df)):
        cls = min(item.diagnosis.split(), key=lambda cls: cls_counts[cls])
        fold_counts = [(f, fold_cls_counts[f, cls]) for f in range(n_folds)]
        min_count = min([count for _, count in fold_counts])
        random.seed(item.Index)
        fold = random.choice([f for f, count in fold_counts
                              if count == min_count])
        folds[item.Index] = fold
        for cls in item.diagnosis.split():
            fold_cls_counts[fold, cls] += 1
    df['fold'] = folds
    return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n-folds', type=int, default=5)
    args = parser.parse_args()
    df = make_folds(n_folds=args.n_folds)
    df.to_csv('../input/folds.csv', index=None)


if __name__ == '__main__':
    main()