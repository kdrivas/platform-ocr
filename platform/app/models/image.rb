class Image < ApplicationRecord

  has_attached_file :text_image, :url  => "/system/:attachment/:id/:style_:filename",
     :path => ":rails_root/public/system/:attachment/:id/:filename"
  validates :text_image, attachment_presence: true
  validates_attachment_content_type :text_image, :content_type => ["image/jpg", "image/jpeg", "image/gif", "image/png"]

  has_many :anns, dependent: :destroy
end
