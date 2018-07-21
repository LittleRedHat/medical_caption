

class Metrics:
    def __init__(self):
        from coco_caption.pycocoevalcap.bleu.bleu import Bleu
        from coco_caption.pycocoevalcap.cider.cider import Cider
        from coco_caption.pycocoevalcap.rouge.rouge import Rouge
        from coco_caption.pycocoevalcap.meteor.meteor import Meteor

        self.bleu = Bleu()
        self.cider = Cider()
        self.rouge = Rouge()
        self.meteor = Meteor()

    def compute_single_score(self, truth, pred):
        '''
        Computer several metrics
        :param truth: <String> the ground truth sentence
        :param pred:  <String> predicted sentence
        :return: score list
        '''
        bleu_gts = {'1': [truth]}
        bleu_res = {'1': [pred]}
        bleu_score = self.bleu.compute_score(bleu_gts, bleu_res)

        rouge_gts = bleu_gts
        rouge_res = bleu_res
        rouge_score = self.rouge.compute_score(rouge_gts, rouge_res)

        return {'BLEU': bleu_score[0], 'ROUGE': rouge_score[0]}

    def compute_set_score(self, truths, preds):
        gts = {k: [v] for k, v in truths.items()}
        res = {k: [v] for k, v in preds.items()}

        bleu_score = self.bleu.compute_score(gts, res)
        rouge_score = self.rouge.compute_score(gts, res)
        cider_score = self.cider.compute_score(gts, res)

        return {'BLEU': bleu_score[0], 'ROUGE': rouge_score[0], 'CIDEr': cider_score[0]}
