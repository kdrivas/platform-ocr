class CreateBboxes < ActiveRecord::Migration[5.1]
  def change
    create_table :bboxes do |t|
      t.float :x0
      t.float :y0
      t.float :x1
      t.float :y1
      t.string :word
      t.timestamps
    end
  end
end
