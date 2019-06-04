import torch

from source.utils.misc import Pack

class emo_generate():
    def __init__(self, model):
        self.model=model

    #  构建和之前一样的结构，方便选择出正确的 emo
    def generate(self,batch_iter, num_batches):
        self.model.eval()
        itoemo = ['NORM', 'POS', 'NEG']
        with torch.no_grad():
            results = []
            for inputs in batch_iter:
                enc_inputs = inputs
                dec_inputs = inputs.num_tgt_input
                enc_outputs=Pack()
                outputs = self.model.forward(enc_inputs, dec_inputs, hidden=None)
                outputs=outputs.logits
                preds = outputs.max(dim=2)
                # news_id = inputs.id
                tgt_raw = inputs.raw_tgt
                preds = preds[1].tolist()

                temp_a_1=[]
                emo_b_1=[]
                temp = []
                tgt_emo = inputs.tgt_emo[0].tolist()
                for  a, b, c in zip( tgt_raw, preds, tgt_emo):
                    # enc_outputs.add(preds=preds, scores=scores, emos=emos, target_emos=temp)
                    # result_batch = enc_outputs.flatten()
                    # results += result_batch
                    a = a[1:]
                    temp_a = []
                    emo_b = []
                    emo_c=[]

                    for i, entity in enumerate(a):
                        temp_a.append(entity)
                        emo_b.append(itoemo[b[i]])
                        emo_c.append(itoemo[c[i]])

                        # tgt_raw=tgt_raw[1:]
                    assert len(temp_a) == len(emo_b)
                    assert len(emo_c) == len(emo_b)
                    temp_a_1.append([temp_a])   #pred1
                    emo_b_1.append([emo_b])   # emo
                    temp.append(emo_c)

                # temp = []
                # tgt_emo = inputs.tgt_emo[0].tolist()
                # for item in tgt_emo:
                #     temp.append([itoemo[x] for x in item])

                # print(emo_b_1)
                # print(temp)

                if hasattr(inputs, 'id') and inputs.id is not None:
                    enc_outputs.add(id=inputs['id'])
                enc_outputs.add(tgt=tgt_raw, preds=temp_a_1, emos=emo_b_1, target_emos=temp)
                result_batch=enc_outputs.flatten()
                results+=result_batch
            return results



