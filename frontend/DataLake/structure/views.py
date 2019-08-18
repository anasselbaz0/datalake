from django.shortcuts import render
from django.core.files.storage import FileSystemStorage
import os
from django.conf import settings
from . import functions as me

# Create your views here.


def index(request):

    if request.method == 'POST':
        sources = request.FILES.getlist('sources')
        for source in sources: 
            fs = FileSystemStorage()
            fs.save(source.name, source)
            print('Source ', source.name, ' uploading ...')
        print('Sources uploaded with success!')
    
    sources = me.data_sources(os.path.join(settings.BASE_DIR, "sources"))

    return render(request, 'structure/index.html', {'sources': sources})


def start(request):
    sources = me.data_sources(os.path.join(settings.BASE_DIR, "sources"))
    
    return render(request, 'structure/start.html', {'sources': sources})