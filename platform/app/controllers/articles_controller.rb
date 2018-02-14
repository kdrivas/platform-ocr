class ArticlesController < ApplicationController

	def index
		respond_to do |format|
			format.html
			format.json
		end
	end

	def new
		@article = Article.new
	end

	def create
		@article = Article.new(article_params)
		@aux = Article.new
		@aux.title = %x[System('python test.py')]
		@aux.text = %x[System('python test.py')]

 		respond_to do |format|
 			if @article.save
 				format.js {render 'show'}	
 			end
 		end
  		
	end

	def show
	end

	private
		def article_params
    		params.require(:article).permit(:title, :text)
  		end
end
