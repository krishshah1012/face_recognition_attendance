from django.contrib import admin
from django.urls import path
from . import views

urlpatterns = [
    path('',views.index,name='index'),
    path('profiles/',views.profiles,name='profiles'),
    path('add_profile/',views.add_profile,name="add_profile"),
    path('edit_profile/<int:id>/',views.edit_profile,name="edit_profile"),
    path('delete_profile/<int:id>/',views.delete_profile,name="delete_profile"),
    path('details/',views.details,name='details'),
    path('reset/',views.reset,name="reset"),
    path('clear_history/',views.clear_history,name="clear_history"),
    path('scan/',views.scan,name="scan"),
    path('ajax/',views.ajax,name="ajax"),
]
