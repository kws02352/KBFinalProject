from django.shortcuts import render
from django.core.files.storage import FileSystemStorage
import e4e
import styleclip
import faceswap

net = e4e.load_e4e_model()
model = styleclip.load_styleclip_model()

# Create your views here.
def index(request):
    context={'a':1}
    return render(request, 'index.html', context)
    
def predictImage(request):
    print("request_sh : ",request)
    print("request.POST.dict()_sh : ",request.POST.dict())
    styleNum = request.POST['style']
    fileobj = request.FILES['filePath']
    fs = FileSystemStorage()
    filePathName = fs.save(fileobj.name, fileobj)
    filePathName = fs.url(filePathName)
    
    e4e.img2latent(net, fileobj.name)
    styleclip.styleImg(model, fileobj.name, styleNum)
    resultFilePathName = '/media/result/' + fileobj.name.split('.')[0] + '.jpg'
    
    faceswap.faceswapMain(filePathName, resultFilePathName)
    
    e4e.check(fileobj.name)
        
    context={'filePathName': filePathName, 'resultFilePathName': resultFilePathName}
    return render(request, 'index.html', context)