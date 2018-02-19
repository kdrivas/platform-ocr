Rails.application.routes.draw do
  get 'welcome/index'
  get 'images/random'

  resources :articles
  resources :images
  #resources :images do
  #  get 'random', on: :collection
  #en
  root 'images#new'
  # For details on the DSL available within this file, see http://guides.rubyonrails.org/routing.html
end
