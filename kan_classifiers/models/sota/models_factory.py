class ModelsFactory:

    def __init__(self, configs):
        self.configs = configs

    def load_model(self, model_name):

        if model_name == 'dinov2-base': # ViT
            return self._get_dinov2_base()
        elif model_name == 'cnn-baseline':
            return
        elif model_name == 'resnet50-baseline':
            return
        elif model_name == 'resnet101-baseline':
            return
        else:
            print('architecture not implemented yet!')
        

    def _get_dinov2_base(self):
        from transformers import AutoImageProcessor, AutoModel # documentation --> https://huggingface.co/facebook/dinov2-base
        processor = AutoImageProcessor.from_pretrained('facebook/dinov2-base')
        model = AutoModel.from_pretrained('facebook/dinov2-base')
        return model, processor