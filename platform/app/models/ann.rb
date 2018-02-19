class Ann < ApplicationRecord
  belongs_to :image
  has_many :bboxes, dependent: :destroy
  enum flag: [:ground, :eval]
end
