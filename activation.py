from fastai.conv_learner import *
import datetime
def activation(file_name):

    layer = 42  # The last Layer
    filter = 64  # will be marked with a vertical line in the plot
    total_filters_in_layer = 512

    class SaveFeatures():
        def __init__(self, module):
            self.hook = module.register_forward_hook(self.hook_fn)
        def hook_fn(self, module, input, output):
            self.features = torch.tensor(output,requires_grad=True)
        def close(self):
            self.hook.remove()

    picture = PIL.Image.open(file_name)
    model = vgg19(pre=True).eval()
    set_trainable(model, False)

    sz = 224
    train_tfms, val_tfms = tfms_from_model(vgg16, sz)
    transformed = val_tfms(np.array(picture)/255)
    activations = SaveFeatures(list(model.children())[layer])
    model(V(transformed)[None]);
    mean_act = [activations.features[0,i].mean().item() for i in range(total_filters_in_layer)]

    plt.figure(figsize=(7,5))
    act = plt.plot(mean_act,linewidth=2.)
    extraticks=[filter]
    ax = act[0].axes
    ax.set_xlim(0,500)
    plt.axvline(x=filter, color='grey', linestyle='--')
    ax.set_xlabel("feature map")
    ax.set_ylabel("mean activation")
    ax.set_xticks([0,200,400] + extraticks)
    #plt.show()
    
    currentDT = datetime.datetime.now()
    out_name = 'mean_activation_layer_'+str(layer)+'_filter_'+str(filter)+str(currentDT)+'.png'
    plt.savefig('./static/images/'+out_name)
    thresh = 0.44
    for i in range(total_filters_in_layer):
        if mean_act[i]>thresh:
            print(i)
    activations.close()
    max = 0
    for i in range(total_filters_in_layer):
      if (mean_act[i] > max):
        max = mean_act[i]
        index = i
    print("MAX", index)
    return out_name
